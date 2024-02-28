from abc import ABC, abstractmethod, abstractproperty

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

    @abstractproperty
    def name(self):
        pass