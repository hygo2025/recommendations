from src.jobs.job_executors.example_job_executor import ExampleJobExecutor
from src.utils.enums import RunnerEventType
from src.utils.enviroment import runner_type


class RunnerJob:
    @staticmethod
    def do(
            runner_type_str: str = None,
    ):
        print(
            f"Running job, runner_type: {runner_type_str}"
        )

        type = RunnerEventType.value_of(runner_type_str)

        executor_classes = {
            "example": ExampleJobExecutor,
        }

        job_executors = [
            cls(runner_type=type)
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
