from typing import Iterable
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.generators.abstract_generator import AbstractGenerator
from tml_decoder.models.abstract_model import AbstractLabelModel
import numpy as np


class BeamSearchModel(AbstractLabelModel):
    def __init__(self, encoder: AbstractEncoder, generator: AbstractGenerator, iter_num=100, beam_width=3) -> None:
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.iter_num = iter_num
        self.beam_width = beam_width

    @property
    def name(self):
        return f"Beam search model, encoder: {self.encoder.name}, generator: {self.generator.name}"

    def get_embedding_to_revert(self, texts: Iterable[str]):
        encoded_texts = np.stack([self.encoder.encode(text) for text in texts])
        return self.encoder.average_embedding(encoded_texts)

    def get_label(self, texts: Iterable[str]) -> str:
        target_embedding = self.get_embedding_to_revert(texts)
        return self.beam_search("", target_embedding, self.iter_num)
    
    def generate_next_states(self, state, target_embedding):
        children = self.generator.generate(state)
        scores = [self.encoder.similarity(self.encoder.encode(child), target_embedding) for child in children]
        return children, scores

    def beam_search(self, initial_state, target_embedding, iter_num):
        best_state = initial_state
        best_score = self.encoder.similarity(
            self.encoder.encode(initial_state), target_embedding
        )

        beam = [(best_state, best_score)]

        for iter in range(iter_num):
            new_beam = []

            for state, _ in beam:
                next_states, next_scores = self.generate_next_states(state, target_embedding)
                top_indices = np.argsort(next_scores)[-self.beam_width:]
                top_idx = top_indices[0]
                if next_scores[top_idx] > best_score:
                    best_score = next_scores[top_idx]
                    best_state = next_states[top_idx]
                new_beam.extend([(next_states[i], next_scores[i]) for i in top_indices])

            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_width]
        
        return best_state
