from tml_decoder.generators.abstract_generator import AbstractGenerator
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch

DEFAULT_DEVICE = None

if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

class GPT2Generator(AbstractGenerator):
    def __init__(self, num_gens=3, device=DEFAULT_DEVICE):
        super().__init__()
        self.num_gens = num_gens
        self.device = device
        self.gpt_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.generator = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(self.device)

    def generate(self, text: str):
        #hack: GPT2LMHeadModel does not support empty token seqs
        if text == "":
            text = self.gpt_tokenizer.eos_token

        inputs = self.gpt_tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.generator(**inputs)
        logits = outputs.logits
        top_k_indices = torch.topk(logits, self.num_gens, dim=-1).indices.flatten()

        if text == self.gpt_tokenizer.eos_token:
            text = ""

        continuations = [text + self.gpt_tokenizer.decode(int(idx)) for idx in top_k_indices]
        return continuations

    def get_tokenizer(self):
        return self.gpt_tokenizer

    
    @property
    def name(self):
        return "GPT2"
