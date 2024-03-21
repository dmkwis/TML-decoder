from typing import List
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from sentence_transformers import SentenceTransformer, util
import numpy as np
from numpy import ndarray

class MiniLMEncoder(AbstractEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
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
    
    @property
    def name(self):
        return "Mini LM"
