from typing import List, Tuple

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2PerplexityEvaluator:
    def __init__(self, model_name: str = "gpt2", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()  # Put the model in evaluation mode
        self.device = device
        self.max_input_size = self.model.config.n_positions

    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> List[float]:
        perplexities = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_perplexities = self._process_batch(batch_texts)
            perplexities.extend(batch_perplexities)
        return perplexities

    def _process_batch(self, texts: List[str]) -> List[float]:
        tokenized_texts = [self.tokenizer.encode(text, return_tensors="pt") for text in texts]
        batch_perplexities = []

        with torch.no_grad():
            for tokens in tokenized_texts:
                tokens = tokens.to(self.device)
                # Handling long texts by chunking them
                chunks = [tokens[0][i : i + self.max_input_size] for i in range(0, tokens.size(1), self.max_input_size - 1)]
                total_loss = 0
                for chunk in chunks:
                    # Adjusting for models that have different input expectations
                    inputs = chunk.unsqueeze(0)
                    outputs = self.model(inputs, labels=inputs)
                    total_loss += outputs.loss.item() * chunk.size(0)

                if tokens.size(1) == 0:
                    perplexity = 0
                else:
                    avg_loss = total_loss / tokens.size(1)
                    perplexity = torch.exp(torch.tensor(avg_loss)).item()
                batch_perplexities.append(perplexity)
        return batch_perplexities


def evaluate_perplexity(
    reference_texts: List[str],
    generated_texts: List[str],
    model_name: str = "gpt2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8,
) -> Tuple[List[float], List[float]]:
    evaluator = GPT2PerplexityEvaluator(model_name=model_name, device=device)
    reference_perplexities = evaluator.calculate_perplexity(reference_texts, batch_size=batch_size)
    generated_perplexities = evaluator.calculate_perplexity(generated_texts, batch_size=batch_size)
    return reference_perplexities, generated_perplexities
