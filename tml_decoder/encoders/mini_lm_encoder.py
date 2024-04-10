from typing import List

import torch
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from sentence_transformers import SentenceTransformer, util
import numpy as np
from numpy import ndarray

from tml_decoder.utils.helper_functions import default_device

class MiniLMEncoder(AbstractEncoder):
    def __init__(self, device=default_device()):
        super().__init__()
        self.device = device
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
    
    def similarity(self, veca: ndarray, vecb: ndarray) -> float:
        return util.dot_score(veca, vecb).item()
    
    def average_embedding(self, embeddings: ndarray) -> ndarray:
        summed = np.sum(embeddings, axis=0)
        return summed / np.linalg.norm(summed)
    
    def encode(self, text: str) -> ndarray:
        result = self.encoder.encode(text)
        if isinstance(result, ndarray):
            return result
        
        raise TypeError("Expected ndarray from self.encoder.encode, got {}".format(type(result)))
    
    def encode_batch(self, texts: List[str]) -> List[ndarray]:
        return self.encoder.encode(texts)
    

    def tokenize_text(self, text: str) -> List[int]:
        return self.encoder.tokenizer.encode(text)

    def get_unused_token(self) -> str:
        return next(
            k for k, v in self.encoder.tokenizer.vocab.items() if "[unused" in k
        )
    
    def get_token_id(self, token: str) -> int:
        return self.encoder.tokenizer.vocab[token]

    def get_characteristic_tensor_for_token_id(self, token_id: int):
        return torch.tensor(
            [i for i in range(len(self.encoder.tokenizer.vocab)) if i != token_id]
        )

    def freeze_weights(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_embedding_weigths(self):
        for idx, p in enumerate(
            self.encoder[0].auto_model.embeddings.word_embeddings.parameters()
        ):
            p.requires_grad = True

    def get_eot_token_id(self) -> int:
        return 102
    
    def get_parameters(self):
        return self.encoder.parameters()
    
    def raw_encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })["sentence_embedding"]
    
    def get_embedding_for_token_id(self, token_id: int):
        return self.encoder[0].auto_model.embeddings.word_embeddings.weight.data[token_id]
    
    def zero_grad_for_embeddings(self, mask: torch.Tensor):
         self.encoder[0].auto_model.embeddings.word_embeddings.weight.grad[mask] = 0
    
    def get_tokenizer_vocab(self):
        return self.encoder.tokenizer.vocab

    @property
    def name(self):
        return "Mini LM"
