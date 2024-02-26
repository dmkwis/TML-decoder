from abc import ABC, abstractmethod

class AbstractGenerator(ABC):
    @abstractmethod
    def generate(self, text: str):
        pass