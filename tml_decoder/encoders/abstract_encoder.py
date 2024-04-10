from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List
from numpy import ndarray
import numpy as np
import torch


class AbstractEncoder(ABC):
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[ndarray]:
        pass

    @abstractmethod
    def encode(self, text: str) -> ndarray:
        pass

    @abstractmethod
    def similarity(self, veca: ndarray, vecb: ndarray) -> float:
        pass

    @abstractmethod
    def average_embedding(self, embeddings: ndarray) -> ndarray:
        pass

    def average_embedding_for_texts(self, texts: List[str]) -> ndarray:
        encoded_batch = np.stack(self.encode_batch(texts))
        return self.average_embedding(encoded_batch)

    @abstractmethod
    def tokenize_text(self, text: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def get_unused_token(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def get_token_id(self, token: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_characteristic_tensor_for_token_id(self, token: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def freeze_weights(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def unfreeze_embedding_weigths(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_eot_token_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def raw_encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def zero_grad_for_embeddings(self, mask: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer_vocab(self):
        raise NotImplementedError

    @abstractmethod
    def get_embedding_for_token_id(self, token_id: int):
        raise NotImplementedError

    @abstractproperty
    def name(self) -> str:
        return "abstract-encoder"
