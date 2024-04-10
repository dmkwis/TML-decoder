from typing import Any, List

import torch
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
import numpy as np

from tml_decoder.utils.helper_functions import default_device


class GtrBaseEncoder(AbstractEncoder):
    def __init__(self, device=default_device()):
        super().__init__()
        self.encoder = SentenceTransformer("sentence-transformers/gtr-t5-base")

    def similarity(self, veca: ndarray, vecb: ndarray) -> float:
        return util.dot_score(veca, vecb).item()

    def average_embedding(self, embeddings: ndarray) -> ndarray:
        res = embeddings.sum(axis=0, keepdims=True)
        return res / np.linalg.norm(res)

    def encode(self, text: str) -> ndarray:
        result = self.encoder.encode(text)

        if isinstance(result, ndarray):
            return result

        raise TypeError(
            "Expected ndarray from self.encoder.encode, got {}".format(type(result))
        )

    def encode_batch(self, texts: List[str]) -> List[ndarray]:
        return self.encoder.encode(texts)

    def tokenize_text(self, text: str) -> List[int]:
        return self.encoder.tokenizer.encode(text)

    def get_unused_token(self) -> str:
        return next(
            k for k, v in self.encoder.tokenizer.vocab.items() if "<extra_id_" in k
        )

    def get_token_id(self, token: str) -> int:
        return self.encoder.tokenizer.vocab[token]

    def get_characteristic_tensor_for_token_id(self, token_id: int):
        return torch.tensor(
            [i for i in range(len(self.encoder.tokenizer.vocab)) if i != token_id]
        )

    def freeze_weights(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_embedding_weigths(self) -> None:
        self.encoder[0]._modules["auto_model"]._modules["encoder"]._modules[
            "embed_tokens"
        ]._parameters["weight"].requires_grad = True

    def get_eot_token_id(self) -> int:
        return 1

    def get_parameters(self):
        return self.encoder.parameters()

    def raw_encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )["sentence_embedding"]

    def zero_grad_for_embeddings(self, mask: torch.Tensor):
        self.encoder[0]._modules["auto_model"]._modules["encoder"]._modules[
            "embed_tokens"
        ]._parameters["weight"].grad[mask] = 0

    def get_tokenizer_vocab(self):
        return self.encoder.tokenizer.vocab

    def get_embedding_for_token_id(self, token_id: int):
        return self.encoder[0]._modules["auto_model"]._modules["encoder"]._modules[
            "embed_tokens"
        ]._parameters["weight"][token_id]

    @property
    def name(self) -> str:
        return "gtr-base"
