from typing import List, Union

from numpy import ndarray
import numpy as np
import torch

from tml_decoder.encoders.abstract_encoder import AbstractEncoder


class MockEncoder(AbstractEncoder):
    def encode_batch(self, texts: List[str]) -> List[ndarray]:
        return [np.array([1, 2, 3]) for _ in range(len(texts))]

    def encode(self, text: str) -> ndarray:
        return np.array([1, 2, 3])

    def similarity(self, veca: ndarray, vecb: ndarray) -> float:
        return 0.9

    def average_embedding(self, embeddings: ndarray) -> ndarray:
        return np.array([2.5, 3.5, 4.5])

    def tokenize_text(self, text: str) -> List[int]:
        return [0, 1, 2]

    def get_unused_token(self) -> str:
        return "[unused1]"

    def get_token_id(self, token: str) -> int:
        return 1

    def get_characteristic_tensor_for_token_id(self, token: str) -> torch.Tensor:
        return torch.tensor([0, 1, 2])

    def freeze_weights(self) -> None:
        pass

    def unfreeze_embedding_weights(self) -> None:
        pass

    def get_eot_token_id(self) -> int:
        return 102

    def get_parameters(self):
        return []

    def raw_encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.1, 0.2, 0.3])

    def zero_grad_for_embeddings(self, mask: torch.Tensor):
        pass

    def get_tokenizer_vocab(self):
        return {}

    def get_embedding_for_token_id(self, token_id: int):
        return torch.tensor([0.1, 0.2, 0.3])

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        return "decoded text"

    def train(self):
        pass

    def eval(self):
        pass

    @property
    def name(self) -> str:
        return "mock-encoder"
