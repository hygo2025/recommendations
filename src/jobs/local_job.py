from src.jobs.runner_job import RunnerJob
from src.utils.enums import RunnerEventType

if __name__ == "__main__":
    RunnerJob.do(
        runner_type_str=RunnerEventType.EXAMPLE.value,
    )
