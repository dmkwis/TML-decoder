from abc import ABC, abstractmethod, abstractproperty
from typing import List


class AbstractLabelModel(ABC):
    @abstractproperty
    def name(self) -> str:
        """Get name of the model"""
        return "abstract-model"

    @abstractmethod
    def get_label(self, texts: List[str]) -> str:
        """
        Abstract method to calculate the label for texts: Iterable[str].
        Be careful not to provide infinite iterables.
        """
        pass
