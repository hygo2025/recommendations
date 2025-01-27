from src.jobs.job_executors.abstract_job_executor import AbstractJobExecutor
from src.runners.example_runner import ExampleRunner

from src.utils.enums import RunnerEventType
from src.utils.spark_session_utils import create_spark_session


class ExampleJobExecutor(AbstractJobExecutor):
    def __init__(
            self,
            runner_type: RunnerEventType,
    ):
        super().__init__( cls_name="Example")
        self.runner_type = runner_type

    def _run(self):
        ExampleRunner(
            spark=create_spark_session("Example"),
        ).run()

    def should_run(self) -> bool:
        return self.runner_type == RunnerEventType.EXAMPLE
