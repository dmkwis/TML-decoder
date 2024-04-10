import abc
from abc import abstractmethod
from typing import List
from numpy import ndarray


class AbstractGuide(abc.ABC):
    @abstractmethod
    def choose_next(self, propositions: List[str], target_embedding: ndarray) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError