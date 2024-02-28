from abc import ABC, abstractmethod, abstractproperty

class AbstractGenerator(ABC):
    @abstractmethod
    def generate(self, text: str):
        pass

    @abstractproperty
    def name(self):
        pass