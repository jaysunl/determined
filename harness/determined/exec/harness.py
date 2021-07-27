import faulthandler
import logging
import os
import pathlib
import sys

import determined as det
from determined import _training, layers, load
from determined.common import experimental
from determined.common.api import certs


def config_logging(worker_process_env: layers.WorkerProcessContext) -> None:
    log_level = logging.DEBUG if worker_process_env.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s:%(levelname)s [%(process)s]: %(message)s"
    )
    logging.getLogger().setLevel(log_level)
    logging.debug("Starting harness.")


def main() -> None:
    if len(sys.argv) != 2:
        print("worker_process_env_path must be provided as a commandline argument", file=sys.stderr)
        sys.exit(1)

    # Load the worker process env.
    worker_process_env_path = pathlib.Path(sys.argv[1])
    worker_process_env = layers.WorkerProcessContext.from_file(worker_process_env_path)

    config_logging(worker_process_env)

    # API code expects credential to be available as an environment variable
    os.environ["DET_ALLOCATION_SESSION_TOKEN"] = worker_process_env.env.det_allocation_token

    # TODO: refactor websocket, data_layer, and profiling to to not use the cli_cert.
    master_url = (
        f"http{'s' if worker_process_env.env.use_tls else ''}://"
        f"{worker_process_env.env.master_addr}:{worker_process_env.env.master_port}"
    )
    certs.cli_cert = certs.default_load(master_url=master_url)

    if worker_process_env.env.experiment_config.debug_enabled():
        faulthandler.dump_traceback_later(30, repeat=True)

    with det._catch_sys_exit():
        try:
            controller = load.prepare_controller(
                worker_process_env.env,
                worker_process_env.rendezvous_info,
                worker_process_env.hvd_config,
            )
        except det.InvalidHP:
            # build a Training API object just to call report_early_exit().
            session = experimental.Session(None, None, None, certs.cli_cert)
            training = _training.Training(
                session,
                int(worker_process_env.env.det_trial_id),
                worker_process_env.env.trial_run_id,
                int(worker_process_env.env.det_experiment_id),
                None,
                None,
            )
            training.report_early_exit(_training.EarlyExitReason.INVALID_HP)
            raise

        controller.run()


if __name__ == "__main__":
    try:
        main()
    except det.InvalidHP:
        logging.info("InvalidHP detected, worker is exiting")
        sys.exit(1)
