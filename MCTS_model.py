from typing import Iterable
from abstract_model import AbstractLabelModel
import common_utils


class MCTSModel(AbstractLabelModel):
    def __init__(
        self,
        encoder: common_utils.AbstractEncoder,
        generator: common_utils.AbstractGenerator,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.generator = generator

    @property
    def name(self):
        return "MCTS model"

    def get_embedding_to_revert(self, texts: Iterable[str]):
        encoded_texts = [self.encoder.encode(text) for text in texts]
        return self.encoder.average_embedding(encoded_texts)
    
    def revert_embedding(self, embedding):
        pass 

    def get_label(self, texts: Iterable[str]) -> str:
        embedding = self.get_embedding_to_revert(texts)
        return self.revert_embedding(embedding)
