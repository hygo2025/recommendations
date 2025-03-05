from src.jobs.job_executors.abstract_job_executor import AbstractJobExecutor
# from src.runners.sar_runner import SarRunner

from src.utils.enums import RunnerEventType
# from src.utils.spark_session_utils import create_spark_session


class SarJobExecutor(AbstractJobExecutor):
    def __init__(
            self,
            runner_type: RunnerEventType,
    ):
        super().__init__( cls_name="Sar")
        self.runner_type = runner_type

    def _run(self):
        # SarRunner(
        #     spark=create_spark_session("Sar"),
        # ).run()
        pass

    def should_run(self) -> bool:
        return self.runner_type == RunnerEventType.SAR
