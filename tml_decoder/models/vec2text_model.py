from typing import Iterable
from tml_decoder.models.abstract_model import AbstractLabelModel
from vec2text import invert_embeddings, load_corrector, models
from sentence_transformers import SentenceTransformer
import torch


class Vec2TextModel(AbstractLabelModel):
    def __init__(self, num_steps=10, device="mps"):
        self.corrector = load_corrector("gtr-base")
        self.encoder = SentenceTransformer(
            "sentence-transformers/gtr-t5-base", device=device
        )
        self.num_steps = num_steps
        self.device = device

    @property
    def name(self):
        return "vec2text model"

    def get_label(self, texts: Iterable[str]) -> str:
        embeddings = self.encoder.encode(texts, convert_to_tensor=True).to(self.device)
        mean_embedding = embeddings.mean(dim=0, keepdim=True)

        [result] = invert_embeddings(
            mean_embedding, self.corrector, num_steps=self.num_steps
        )

        return result
