from abc import ABC, abstractmethod, abstractproperty

class AbstractGenerator(ABC):
    @abstractmethod
    def generate(self, text: str):
        raise NotImplementedError

    @abstractproperty
    def name(self):
        raise NotImplementedError