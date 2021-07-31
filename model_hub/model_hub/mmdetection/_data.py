import functools
import logging
import math
import os
from typing import Any, Iterator, Tuple

import filelock
import mmcv
import mmcv.parallel
import mmdet.datasets
import numpy as np
import torch
import torch.utils.data as torch_data

import determined.pytorch as det_torch


class GroupSampler(torch.utils.data.Sampler):
    """
    Modifies DistributedGroupSampler from
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/samplers/group_sampler.py
    to work with our Dataloader which automatically handles sharding for distributed training.
    """

    def __init__(
        self,
        dataset: torch_data.Dataset,
        samples_per_gpu: int,
        num_replicas: int,
    ):
        """
        Arguments:
            dataset: Dataset used for sampling.
            num_replicas (optional): Number of processes participating in
                distributed training.
        """
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas

        assert hasattr(self.dataset, "flag")
        self.flag = self.dataset.flag  # type: ignore
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for size in self.group_sizes:
            self.num_samples += (
                int(math.ceil(size * 1.0 / self.samples_per_gpu / self.num_replicas))
                * self.samples_per_gpu
            )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[Any]:
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size)))].tolist()
                extra = int(
                    math.ceil(size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[: extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j]
            for i in list(torch.randperm(len(indices) // self.samples_per_gpu))
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]

        return iter(indices)

    def __len__(self) -> int:
        return self.total_size


def maybe_download_ann_file(cfg: mmcv.Config) -> None:
    if "dataset" in cfg:
        dataset = cfg.dataset
    else:
        dataset = cfg
    ann_dir = "/".join(dataset.ann_file.split("/")[0:-1])
    os.makedirs(ann_dir, exist_ok=True)
    lock = filelock.FileLock(dataset.ann_file + ".lock")

    with lock:
        if not os.path.isfile(dataset.ann_file):
            try:
                assert (
                    dataset.pipeline[0].type == "LoadImageFromFile"
                ), "First step of dataset.pipeline is not LoadImageFromFile."
                file_client_args = dataset.pipeline[0].file_client_args
                file_client = mmcv.FileClient(**file_client_args)
                ann_bytes = file_client.get(dataset.ann_file)
                logging.info(f"Downloading annotation file using {file_client.backend} backend.")
                with open(dataset.ann_file, "wb") as f:
                    f.write(ann_bytes)
            except Exception as e:
                logging.error(
                    f"Could not download missing annotation file.  Encountered {e}."
                    f"Please make sure it is available at the following path {dataset.ann_file}."
                )


class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __getattr__(self, item: Any) -> Any:
        return getattr(self.dataset, item)

    def __getitem__(self, idx: int) -> Any:
        sample = self.dataset[idx]
        if "idx" not in sample["img_metas"][0].data:
            sample["img_metas"][0].data["idx"] = idx
        return sample

    def __len__(self) -> int:
        return self.dataset.__len__()  # type: ignore


def build_dataloader(
    cfg: mmcv.Config,
    context: det_torch.PyTorchTrialContext,
    split: "str",
    num_samples_per_gpu: int,
    num_replicas: int,
    num_workers: int,
    shuffle: bool,
) -> Tuple[torch_data.Dataset, det_torch.DataLoader]:
    """
    Build the dataset and dataloader according to cfg and sampler parameters.

    Arguments:
        cfg: mmcv cfg
        num_samples_per_gpu: samples per gpu
        num_replicas: total number of slots for distributed  training
        num_workers: number of workers to use for data loading
        shuffle: whether to shuffle indices for data loading
        test_mode: if true then annotations are not loaded
    Returns:
        dataset and dataloader
    """
    maybe_download_ann_file(cfg)

    test_mode = False if split == "train" else True
    dataset = mmdet.datasets.build_dataset(cfg, {"test_mode": test_mode})
    if test_mode:
        dataset = DatasetWithIndex(dataset)
    sampler = GroupSampler(dataset, num_samples_per_gpu, num_replicas) if shuffle else None
    return dataset, det_torch.DataLoader(
        dataset,
        batch_size=num_samples_per_gpu,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=functools.partial(mmcv.parallel.collate, samples_per_gpu=num_samples_per_gpu),
        pin_memory=False,
        worker_init_fn=functools.partial(
            mmdet.datasets.builder.worker_init_fn,
            seed=context.get_trial_seed(),
            rank=context.distributed.get_rank(),
            num_workers=num_workers,
        ),
    )
