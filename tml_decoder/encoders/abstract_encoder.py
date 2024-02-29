from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List
from numpy import ndarray
import numpy as np

class AbstractEncoder(ABC):
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[ndarray]:
        pass

    @abstractmethod
    def encode(self, text: str) -> ndarray:
        pass

    @abstractmethod
    def similarity(self, veca: ndarray, vecb: ndarray) -> ndarray:
        pass

    @abstractmethod
    def average_embedding(self, embeddings: ndarray) -> ndarray:
        pass

    def average_embedding_for_texts(self, texts: List[str]) -> ndarray:
        encoded_batch = np.stack(self.encode_batch(texts))
        return self.average_embedding(encoded_batch)


    @abstractproperty
    def name(self) -> str:
        return "abstract-encoder"