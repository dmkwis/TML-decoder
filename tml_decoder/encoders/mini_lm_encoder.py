from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from sentence_transformers import SentenceTransformer
import numpy as np


class MiniLMEncoder(AbstractEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def similarity(self, veca, vecb):
        return veca@vecb
    
    def average_embedding(self, embeddings):
        stacked = np.stack(embeddings)
        return np.sum(stacked, axis=0)
    
    def encode(self, text: str) -> np.ndarray:
        result = self.encoder.encode(text)
        if isinstance(result, np.ndarray):
            return result
        
        raise TypeError("Expected ndarray from self.encoder.encode, got {}".format(type(result)))
    
    @property
    def name(self):
        return "Mini LM"
