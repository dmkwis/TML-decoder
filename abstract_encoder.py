from abc import ABC, abstractmethod

class AbstractEncoder(ABC):
    @abstractmethod
    def encode(self, text: str):
        pass

    @abstractmethod
    def similarity(self, veca, vecb):
        pass

    @abstractmethod
    def average_embedding(self, embeddings):
        pass