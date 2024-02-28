from typing import Any
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from sentence_transformers import SentenceTransformer
from numpy import ndarray


class GtrBaseEncoder(AbstractEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = SentenceTransformer("sentence-transformers/gtr-t5-base")
    
    def similarity(self, veca: ndarray, vecb: ndarray) -> ndarray:
        return veca @ vecb
    
    def average_embedding(self, embeddings: ndarray) -> ndarray:
        return embeddings.mean(axis=0, keepdims=True)
    
    def encode(self, text: str) -> ndarray:
        result = self.encoder.encode(text)
    
        if isinstance(result, ndarray):
              return result
        
        raise TypeError("Expected ndarray from self.encoder.encode, got {}".format(type(result)))
    
    @property
    def name(self) -> str:
        return "gtr-base"
