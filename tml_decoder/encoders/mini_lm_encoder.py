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
        emb_sum = np.sum(stacked, axis=0)
        return emb_sum/np.linalg.norm(emb_sum)
    
    def encode(self, text: str):
        return self.encoder.encode(text)
    
    @property
    def name(self):
        return "Mini LM"

