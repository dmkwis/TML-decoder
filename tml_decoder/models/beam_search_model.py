from typing import Iterable
import numpy as np
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.generators.abstract_generator import AbstractGenerator
from tml_decoder.models.abstract_model import AbstractLabelModel

class BeamSearchModel(AbstractLabelModel):
    def __init__(
        self,
        encoder: AbstractEncoder,
        generator: AbstractGenerator,
        iter_num=100,
        beam_width=5,
        min_result_len=3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.iter_num = iter_num
        self.beam_width = beam_width
        self.min_result_len = min_result_len

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
        encodings = self.encoder.encode_batch(children)
        scores = [
            self.encoder.similarity(encoding, target_embedding)
            for encoding in encodings
        ]
        return children, scores

    def beam_search(self, initial_state, target_embedding, iter_num):
        best_state = initial_state
        best_score = float("-inf")

        beam = [(initial_state, best_score)]

        for _ in range(iter_num):
            print(beam)
            new_beam = []
            for state, _ in beam:
                next_states, next_scores = self.generate_next_states(state, target_embedding)
                new_beam.extend(zip(next_states, next_scores))

            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:self.beam_width]
            if new_beam[0][1] > best_score and len(new_beam[0][0]) >= self.min_result_len:
                best_state, best_score = new_beam[0]

            beam = new_beam
        print("bs: ", best_state, len(best_state))
        return best_state
