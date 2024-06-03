from abc import ABC, abstractmethod, abstractproperty
from typing import List, Mapping


class AbstractGenerator(ABC):
    def __init__(self):
        self.lazy_batch_args = []

    def add_to_lazy_batch_args(self, text: str) -> None:
        self.lazy_batch_args.append(text)

    def calculate_perplexity_lazy_batch(self) -> Mapping[str, float]:
        return map(zip(self.lazy_batch_args, self.calculate_perplexity(self.lazy_batch_args)))

    def reset_lazy_batch_args(self) -> None:
        self.lazy_batch_args = []

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
