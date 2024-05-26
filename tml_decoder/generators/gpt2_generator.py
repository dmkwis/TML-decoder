from typing import List

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from tml_decoder.generators.abstract_generator import AbstractGenerator
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
            continuations = [continuation.replace(self.gpt_tokenizer.eos_token, "", 1) for continuation in continuations]

        return continuations

    def get_tokenizer(self):
        return self.gpt_tokenizer

    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> List[float]:
        all_perplexities = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch_texts = texts[i * batch_size : (i + 1) * batch_size]
            batch_tokenized = self.gpt_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.generator(**batch_tokenized, labels=batch_tokenized["input_ids"])
                batch_loss = outputs.loss
                perplexities = torch.exp(batch_loss)
                all_perplexities.extend(perplexities.cpu().tolist())

        return all_perplexities

    @property
    def name(self):
        return "GPT2"
