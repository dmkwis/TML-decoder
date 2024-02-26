import torch
from abstract_encoder import AbstractEncoder
from sentence_transformers import SentenceTransformer


class MiniLMEncoder(AbstractEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def similarity(self, veca, vecb):
        return veca@vecb
    
    def average_embedding(self, embeddings):
        stacked = torch.stack(embeddings)
        emb_sum = torch.sum(stacked, dim=0)
        return torch.nn.functional.normalize(emb_sum, p=2)
    
    def encode(self, text: str):
        return self.encoder.encode(text)

