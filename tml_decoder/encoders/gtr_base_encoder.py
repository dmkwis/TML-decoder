from typing import Any, List
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
import numpy as np


class GtrBaseEncoder(AbstractEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = SentenceTransformer("sentence-transformers/gtr-t5-base")
    
    def similarity(self, veca: ndarray, vecb: ndarray) -> float:
        return util.dot_score(veca, vecb).item()
    
    def average_embedding(self, embeddings: ndarray) -> ndarray:
        res = embeddings.sum(axis=0, keepdims=True)
        return res/np.linalg.norm(res)
    
    def encode(self, text: str) -> ndarray:
        result = self.encoder.encode(text)
    
        if isinstance(result, ndarray):
              return result
        
        raise TypeError("Expected ndarray from self.encoder.encode, got {}".format(type(result)))
    
    def encode_batch(self, texts: List[str]) -> List[ndarray]:
        return self.encoder.encode(texts)
    
    @property
    def name(self) -> str:
        return "gtr-base"

