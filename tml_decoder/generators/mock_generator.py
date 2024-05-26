from typing import List

from tml_decoder.generators.abstract_generator import AbstractGenerator


# Mock AbstractGenerator
class MockGenerator(AbstractGenerator):
    def generate(self, text: str):
        return "generated text"

    def get_tokenizer(self):
        return "tokenizer"

    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> List[float]:
        return [1.0] * len(texts)

    @property
    def name(self):
        return "mock-generator"
