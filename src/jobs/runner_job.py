import sys

import numpy as np
import pandas as pd
import scipy

from src.jobs.job_executors.sas_rec_job_executor import SasRecJobExecutor
from src.utils.logger import Logger
from src.utils.enums import RunnerEventType
from src.utils.enviroment import runner_type


class RunnerJob:
    @staticmethod
    def do(
            runner_type_str: str = None,
    ):
        logger = Logger.get_logger(name="RunnerJob")
        logger.info(f"-"*70)
        logger.info(f"Pandas version: {pd.__version__}")
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"System version: {sys.version}")
        logger.info(f"SciPy version: {scipy.__version__}")
        logger.info(f"-"*70)

        logger.info(f"Running job, runner_type: {runner_type_str}")

        executor_classes = {
            "sas_rec": SasRecJobExecutor,
        }

        job_executors = [
            cls(runner_type=RunnerEventType.value_of(runner_type_str))
            for cls in executor_classes.values()
        ]

        for job_executor in job_executors:
            if job_executor.should_run():
                job_executor.run()
                return


if __name__ == "__main__":
    RunnerJob.do(
        runner_type_str=runner_type(),
    )
