from src.jobs.job_executors.abstract_job_executor import AbstractJobExecutor
from src.runners.sas_rec_runner import SasRecRunner

from src.utils.enums import RunnerEventType


class SasRecJobExecutor(AbstractJobExecutor):
    def __init__(
            self,
            runner_type: RunnerEventType,
    ):
        super().__init__( cls_name="Sar")
        self.runner_type = runner_type

    def _run(self):
        SasRecRunner().run()

    def should_run(self) -> bool:
        return self.runner_type == RunnerEventType.SAS_REC
