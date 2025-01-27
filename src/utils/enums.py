from enum import Enum

class RunnerEventType(Enum):
    EXAMPLE = "example"

    @classmethod
    def value_of(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"{value} is not a valid {cls.__name__}")