from abc import ABC, abstractmethod, abstractproperty

class AbstractGenerator(ABC):
    @abstractmethod
    def generate(self, text: str):
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self):
        raise NotImplementedError

    @abstractproperty
    def name(self):
        raise NotImplementedError