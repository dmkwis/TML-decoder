from abc import ABC, abstractmethod, abstractproperty
from typing import List


class AbstractGenerator(ABC):
    @abstractmethod
    def generate(self, text: str):
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def calculate_perplexity(self, texts: List[str], batch_size: int) -> List[float]:
        raise NotImplementedError

    @abstractproperty
    def name(self):
        raise NotImplementedError
