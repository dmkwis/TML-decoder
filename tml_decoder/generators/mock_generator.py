from tml_decoder.generators.abstract_generator import AbstractGenerator


# Mock AbstractGenerator
class MockGenerator(AbstractGenerator):
    def generate(self, text: str):
        return "generated text"

    def get_tokenizer(self):
        return "tokenizer"

    def calculate_perplexity(self, text: str):
        return 5.0

    @property
    def name(self):
        return "mock-generator"
