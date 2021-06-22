import os
import pathlib
import pickle
import time
from typing import Any, Dict, Optional

import determined as det
from determined import horovod, layers, util, workload

from determined.common.api import certs
from determined.common.experimental.session import Session


class NoopTrialController(det.TrialController):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.wlsq = None
        if self.workloads is None:
            session = Session(None, None, None, certs.cli_cert)

            self.workloads, self.wlsq = layers.make_compatibility_workloads(
                session, self.env, self.context.distributed
            )

        if self.env.latest_checkpoint is not None:
            with self._generic._download_initial_checkpoint(self.env.latest_checkpoint) as path:
                self.load(pathlib.Path(path))

    def pre_execute_hook(env: det.EnvContext, hvd_config: horovod.HorovodContext) -> Any:
        if hvd_config.use:
            # DistributedContext needs a working hvd object.
            from determined import horovod
            horovod.hvd.require_horovod_type("torch", "for NoopTrialController")
            horovod.hvd.init()

    def from_trial(trial_inst: det.Trial, *args: Any, **kwargs: Any) -> det.TrialController:
        return NoopTrialController(*args, **kwargs)

    def run(self) -> None:
        for w, response_func in self.workloads:
            start_time = self._generic._current_timestamp()
            try:
                if w.kind == workload.Workload.Kind.RUN_STEP:
                    response = util.wrap_metrics(
                        self.train_for_step(
                            w.step_id, w.num_batches
                        ),  # type: workload.Response
                        False,
                        False,
                        False,
                    )
                    response = self._generic._after_training(w, start_time, response)
                elif w.kind == workload.Workload.Kind.COMPUTE_VALIDATION_METRICS:
                    response = util.wrap_metrics(
                        self.compute_validation_metrics(w.step_id),
                        False,
                        False,
                        False,
                    )
                    searcher_metric = self.env.experiment_config.get_searcher_metric()
                    response = self._generic._after_validation(
                        w, start_time, searcher_metric, response
                    )
                elif w.kind == workload.Workload.Kind.CHECKPOINT_MODEL:
                    with self._generic._storage_mgr.store_path() as (storage_id, path):
                        self.save(pathlib.Path(path))
                        response = {}
                        response = self._generic._after_checkpoint(
                            w,
                            start_time,
                            storage_id,
                            path,
                            response,
                        )
                else:
                    raise AssertionError("Unexpected workload: {}".format(w.kind))

            except det.errors.SkipWorkloadException:
                response = workload.Skipped()

            response_func(response)

    # Methods implemented by AF-specific subclasses.
    def train_for_step(self, step_id: int, batches_per_step: int) -> Dict[str, Any]:
        metrics = {"loss": 0. if (step_id%2) else 1.}
        time.sleep(1)
        return det.util.make_metrics(
            self.batch_size * batches_per_step, [metrics] * batches_per_step
        )

    def compute_validation_metrics(self, step_id: int) -> Dict[str, Any]:
        metrics = {"error": 1. if (step_id%2) else 0.}
        return {"validation_metrics": metrics, "num_inputs": 1}

    def save(self, path: pathlib.Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with path.joinpath("asdf") as f:
            pass

        wlsq_path = path.joinpath("workload_sequencer.pkl")
        if self.wlsq is not None:
            with wlsq_path.open("wb") as f:
                pickle.dump(self.wlsq.get_state() ,f)

    def load(self, path: pathlib.Path) -> None:
        wlsq_path = path.joinpath("workload_sequencer.pkl")
        if self.wlsq is not None and wlsq_path.exists():
            with wlsq_path.open("rb") as f:
                self.wlsq.load_state(pickle.load(f))


class NoopTrial(det.Trial):
    trial_controller_class = NoopTrialController

    def __init__(self, context: det.TrialContext) -> None:
        # for k, v in os.environ.items():
        #     print(f"{k}={v}")
        pass
