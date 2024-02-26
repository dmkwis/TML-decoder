from abstract_generator import AbstractGenerator
from transformers import pipeline, set_seed


class GPT2Generator(AbstractGenerator):
    def __init__(self, step_size, num_gens):
        super().__init__()
        set_seed(42)
        self.generator = pipeline("text-generation", model="gpt2")
        self.step_size = step_size
        self.num_gens = num_gens

    def generate(self, text: str):
        results = self.generator(
            text,  # sentence to continue
            max_length=len(text) + self.step_size,  # only adding step size tokens
            num_return_sequences=self.num_gens,
            pad_token_id=50256,  # default value for pad token in gpt2
        )
        return [res["generated_text"] for res in results] # for now returning only generated sentences, probabilities in the future?
    
    @property
    def name(self):
        return "GPT2"
