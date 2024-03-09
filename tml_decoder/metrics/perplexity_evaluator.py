from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPT2PerplexityEvaluator:
    def __init__(self, model_name='gpt2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()  # Put the model in evaluation mode
        self.device = device
        self.max_input_size = self.model.config.n_positions

    def calculate_perplexity(self, texts, batch_size=8):
        perplexities = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_perplexities = self._process_batch(batch_texts)
            perplexities.extend(batch_perplexities)
        return perplexities

    def _process_batch(self, texts):
        tokenized_texts = [self.tokenizer.encode(text, return_tensors="pt") for text in texts]
        batch_perplexities = []

        with torch.no_grad():
            for tokens in tokenized_texts:
                tokens = tokens.to(self.device)
                # Handling long texts by chunking them
                chunks = [tokens[0][i:i+self.max_input_size] for i in range(0, tokens.size(1), self.max_input_size-1)]
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

# Example usage
if __name__ == "__main__":
    texts = ["This is a test." * 100, "Another example." * 100]  # Example long texts
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    evaluator = GPT2PerplexityEvaluator(device=device)
    perplexities = evaluator.calculate_perplexity(texts)
    for text, perplexity in zip(texts, perplexities):
        print(f"Perplexity: {perplexity}")