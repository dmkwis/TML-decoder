from abc import ABC, abstractmethod, abstractproperty
from typing import Iterable

class AbstractModel(ABC):
    @abstractmethod
    def get_label(self, texts: Iterable[str]) -> str:
        """
            Abstract method to calculate the label for texts: Iterable[str].
            Be careful not to provide infinite iterables.
        """
        pass

    @abstractproperty
    def name(self) -> str:
        """ Get name of the method """
        pass