from src.jobs.job_executors.abstract_job_executor import AbstractJobExecutor
# from src.runners.sar_runner import SarRunner
from src.runners.sas_runner import SasRunner

from src.utils.enums import RunnerEventType


class SasJobExecutor(AbstractJobExecutor):
    def __init__(
            self,
            runner_type: RunnerEventType,
    ):
        super().__init__( cls_name="Sar")
        self.runner_type = runner_type

    def _run(self):
        SasRunner().run()

    def should_run(self) -> bool:
        return self.runner_type == RunnerEventType.SAS
