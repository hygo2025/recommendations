from abc import ABC, abstractmethod

from src.utils.logger import Logger


class AbstractJobExecutor(ABC):

    def __init__(self, cls_name: str):
        self.cls_name = cls_name
        self.logger = Logger.get_logger(name=cls_name)


    def run(self):
        self._run()


    @abstractmethod
    def _run(self):
        raise NotImplementedError("This method should be overridden in derived classes")

    @abstractmethod
    def should_run(self) -> bool:
        raise NotImplementedError("This method should be overridden in derived classes")
