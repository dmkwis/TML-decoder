from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List
from numpy import ndarray

class AbstractEncoder(ABC):
    @abstractmethod
    def encode(self, text: str) -> ndarray:
        pass

    @abstractmethod
    def similarity(self, veca: ndarray, vecb: ndarray) -> ndarray:
        pass

    @abstractmethod
    def average_embedding(self, embeddings: ndarray) -> ndarray:
        pass

    @abstractproperty
    def name(self) -> str:
        return "abstract-encoder"