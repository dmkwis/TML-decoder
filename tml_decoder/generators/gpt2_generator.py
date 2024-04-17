from tml_decoder.generators.abstract_generator import AbstractGenerator
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch

from tml_decoder.utils.helper_functions import default_device


class GPT2Generator(AbstractGenerator):
    def __init__(self, num_gens=3, device=default_device()):
        super().__init__()
        self.num_gens = num_gens
        self.device = device
        self.gpt_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.generator = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(self.device)
        self.generator.eval()
        
    def generate(self, text: str):
        if text == "":
            text = self.gpt_tokenizer.eos_token

        inputs = self.gpt_tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.generator(**inputs)
        logits = outputs.logits

        # Only consider the last token's logits for top-k predictions
        last_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        _, top_k_indices = torch.topk(last_token_logits, self.num_gens, dim=-1)  # Shape: [batch_size, num_gens]
        
        top_k_indices = top_k_indices.flatten()

        continuations = [text + self.gpt_tokenizer.decode(int(idx)) for idx in top_k_indices]

        # If the initial text was empty and replaced with EOS token, remove it from the output
        if text == self.gpt_tokenizer.eos_token:
            continuations = [continuation.replace(self.gpt_tokenizer.eos_token, '', 1) for continuation in continuations]

        return continuations

    def get_tokenizer(self):
        return self.gpt_tokenizer

    def calculate_perplexity(self, text: str) -> float:
        tokens = self.gpt_tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Chunking for long texts
            chunks = [tokens[0][i:i+self.generator.config.n_positions] for i in range(0, tokens.size(1), self.generator.config.n_positions-1)]
            total_loss = 0
            for chunk in chunks:
                inputs = chunk.unsqueeze(0)
                outputs = self.generator(inputs, labels=inputs)
                total_loss += outputs.loss.item() * chunk.size(0)

            if tokens.size(1) > 0:
                avg_loss = total_loss / tokens.size(1)
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
            else:
                perplexity = float('inf')  # or some suitable default for empty text
            return perplexity

    
    @property
    def name(self):
        return "GPT2"
