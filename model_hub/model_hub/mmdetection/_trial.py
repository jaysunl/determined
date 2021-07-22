"""
Determined training loop for mmdetection
mmdetection: https://github.com/open-mmlab/mmdetection.
"""

import logging
from typing import Any, Dict, List

import attrdict
import mmcv
import mmcv.parallel
import mmcv.runner
import mmdet.core
import mmdet.datasets
import mmdet.models
import numpy as np
import torch

import determined.pytorch as det_torch
import model_hub.mmdetection
from determined.horovod import hvd


class MMDetTrial(det_torch.PyTorchTrial):
    def __init__(self, context: det_torch.PyTorchTrialContext) -> None:
        self.context = context
        self.hparams = attrdict.AttrDict(context.get_hparams())
        self.data_config = attrdict.AttrDict(context.get_data_config())
        self.cfg = self.build_mmcv_config()

        self.model = mmdet.models.build_detector(self.cfg.model)
        self.convert_syncbn_model(self.model)
        self.model.init_weights()

        if self.hparams.use_pretrained:
            ckpt_path, ckpt = model_hub.mmdetection.get_pretrained_ckpt_path(
                "/tmp", self.hparams.config_file
            )
            if ckpt_path is not None:
                logging.info("Loading from pretrained weights.")
                if "state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["state_dict"])
                else:
                    self.model.load_state_dict(ckpt)
        self.model = self.context.wrap_model(self.model)

        self.optimizer = self.context.wrap_optimizer(
            mmcv.runner.build_optimizer(self.model, self.cfg.optimizer)
        )

        if self.hparams.use_apex_amp:
            self.model, self.optimizer = self.context.configure_apex_amp(
                models=self.model,
                optimizers=self.optimizer,
            )

        self.clip_grads_fn = (
            lambda x: torch.nn.utils.clip_grad_norm_(x, self.hparams.clip_grads_norm)
            if self.hparams.clip_grads
            else None
        )

    def build_mmcv_config(self) -> mmcv.Config:
        cfg = mmcv.Config.fromfile(self.hparams.config_file)
        cfg.data.val.test_mode = True

        if self.data_config.file_client_args is not None:
            model_hub.mmdetection.sub_backend(self.data_config.file_client_args, cfg)
        if self.hparams.merge_config is not None:
            override_config = mmcv.Config.fromfile(self.hparams.merge_config)
            new_config = mmcv.Config._merge_a_into_b(override_config, cfg._cfg_dict)
            cfg = mmcv.Config(new_config, cfg._text, cfg._filename)

        if "override_config_fields" in self.hparams:
            cfg.merge_from_dict(self.hparams.override_config_fields)

        cfg.data.val.pipeline = mmdet.datasets.replace_ImageToTensor(cfg.data.val.pipeline)
        cfg.data.test.pipeline = mmdet.datasets.replace_ImageToTensor(cfg.data.test.pipeline)

        cfg.dump("./final_config.{}".format(cfg._filename.split(".")[-1]))
        logging.info(cfg)
        return cfg

    def build_callbacks(self) -> Dict[str, det_torch.PyTorchCallback]:
        self.lr_updater = None
        callbacks = {
            "eval_callback": model_hub.mmdetection.EvalCallback(self.context, self.model)
        }  # type: Dict[str, det_torch.PyTorchCallback]
        if "lr_config" in self.cfg:
            logging.info("Adding lr updater callback.")
            self.lr_updater = model_hub.mmdetection.LrUpdaterCallback(
                self.context, lr_config=self.cfg.lr_config
            )
            callbacks["lr_updater"] = self.lr_updater
        return callbacks

    def convert_syncbn_model(self, module: torch.nn.Module) -> Any:
        mod = module
        if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
            mod = hvd.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
            )
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var
            mod.num_batches_tracked = module.num_batches_tracked
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()
        for name, child in module.named_children():
            mod.add_module(name, self.convert_syncbn_model(child))
        del module
        return mod

    def train_batch(self, batch: Any, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        if self.lr_updater is not None:
            self.lr_updater.on_batch_start()
        batch = {key: batch[key].data[0] for key in batch}
        losses = self.model.forward_train(**batch)
        loss, log_vars = self.model._parse_losses(losses)
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer, clip_grads=self.clip_grads_fn)

        lr = self.optimizer.param_groups[0]["lr"]
        metrics = {"loss": loss, "lr": lr}
        metrics.update(log_vars)
        return metrics

    def evaluate_batch(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        batch = {key: batch[key][0].data for key in batch}
        with torch.no_grad():  # type: ignore
            result = self.model(return_loss=False, rescale=True, **batch)
        if isinstance(result[0], tuple):
            result = [
                (bbox_results, mmdet.core.encode_mask_results(mask_results))
                for bbox_results, mask_results in result
            ]
        self.reducer.update(([b["idx"] for b in batch["img_metas"][0]], result))  # type: ignore
        return {}

    def build_training_data_loader(self) -> det_torch.DataLoader:
        dataset, dataloader = model_hub.mmdetection.build_dataloader(
            self.cfg.data.train,
            self.context,
            "train",
            self.context.get_per_slot_batch_size(),
            self.context.distributed.get_size(),
            self.cfg.data.workers_per_gpu,
            True,
        )
        self.model.CLASSES = dataset.CLASSES  # type: ignore
        return dataloader

    def build_validation_data_loader(self) -> det_torch.DataLoader:
        dataset, dataloader = model_hub.mmdetection.build_dataloader(
            self.cfg.data.val,
            self.context,
            "validation",
            self.hparams.validation_batch_size_per_gpu,
            self.context.distributed.get_size(),
            self.cfg.data.workers_per_gpu,
            False,
        )

        def evaluate_fn(results: List[Any]) -> Any:
            # Determined's distributed batch sampler interleaves shards on each GPU slot so
            # sample i goes to worker with rank i % world_size.  Therefore, we need to re-sort
            # all the samples once we gather the predictions before computing the validation metric.
            inds, results = zip(*results)
            inds = [ind for sub_ind in inds for ind in sub_ind]
            results = [res for result in results for res in result]
            sorted_inds = np.argsort(inds)
            results = [results[i] for i in sorted_inds]

            eval_kwargs = self.cfg.evaluation

            for key in ["interval", "tmpdir", "start", "gpu_collect"]:
                eval_kwargs.pop(key, None)

            metrics = dataset.evaluate(results, **eval_kwargs)  # type: ignore
            if not len(metrics):
                return {"bbox_mAP": 0}
            return metrics

        self.reducer = self.context.wrap_reducer(
            evaluate_fn, for_training=False, for_validation=True
        )
        return dataloader

    def get_batch_length(self, batch: Any) -> int:
        if isinstance(batch["img"], mmcv.parallel.data_container.DataContainer):
            length = len(batch["img"].data[0])
        else:
            # The validation data has a different format so we have separate handling below.
            length = len(batch["img"][0].data[0])
        return length

    def to_device(self, context: det_torch.PyTorchTrialContext, batch: Any) -> Dict[str, Any]:
        new_data = {}
        for k, item in batch.items():
            if isinstance(item, mmcv.parallel.data_container.DataContainer) and not item.cpu_only:
                new_data[k] = mmcv.parallel.data_container.DataContainer(
                    context.to_device(item.data),
                    item.stack,
                    item.padding_value,
                    item.cpu_only,
                    item.pad_dims,
                )
            # The validation data has a different format so we have separate handling below.
            elif (
                isinstance(item, list)
                and len(item) == 1
                and isinstance(item[0], mmcv.parallel.data_container.DataContainer)
                and not item[0].cpu_only
            ):
                new_data[k] = [
                    mmcv.parallel.data_container.DataContainer(
                        context.to_device(item[0].data),
                        item[0].stack,
                        item[0].padding_value,
                        item[0].cpu_only,
                        item[0].pad_dims,
                    )
                ]
            else:
                new_data[k] = item
        return new_data


"""
if __name__ == "__main__":
    from determined.common import util

    config_file = "./maskrcnn.yaml"
    with open(config_file, "rb") as f:
        config = util.safe_load_yaml_with_exceptions(f)
    context = det_torch.PyTorchTrialContext.from_config(config)
    trial = MMDetTrial(context)
    trial.model = trial.model.cuda()
    data_loader = trial.build_training_data_loader()
    data_loader = data_loader.get_data_loader()
    for i, batch in enumerate(data_loader):
        batch = trial.to_device(context, batch)
        trial.train_batch(batch, 0, i)
"""
