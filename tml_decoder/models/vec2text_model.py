from typing import List

from numpy import stack
from torch import from_numpy
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.models.abstract_model import AbstractLabelModel
from vec2text import invert_embeddings, load_pretrained_corrector


class Vec2TextModel(AbstractLabelModel):
    def __init__(self, encoder: AbstractEncoder, num_steps=10, device="mps"):
        assert encoder.name == "gtr-base", "Vec2Text supports only gtr-base encoder"

        self.corrector = load_pretrained_corrector("gtr-base")
        self.encoder = encoder
        self.num_steps = num_steps
        self.device = device

    @property
    def name(self):
        return "vec2text model"

    def get_label(self, texts: List[str]) -> str:
        embeddings_list = [self.encoder.encode(text) for text in texts]
        embeddings = stack(embeddings_list)
        mean_embedding = from_numpy(self.encoder.average_embedding(embeddings)).to(self.device)

        [result] = invert_embeddings(
            mean_embedding, self.corrector, num_steps=self.num_steps
        )

        return result
