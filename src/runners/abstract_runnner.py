from abc import ABC, abstractmethod


class AbstractRunner(ABC):

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError("This method should be overridden in derived classes")
