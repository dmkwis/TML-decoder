from typing import Dict, List

from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.generators.abstract_generator import AbstractGenerator


class Metrics:
    def __init__(self, encoder: AbstractEncoder, generator: AbstractGenerator, batch_size: int = 8):
        self.encoder = encoder
        self.generator = generator
        self.batch_size = batch_size

    def calculate_cosine_similarity(self, true_labels: List[str], generated_labels: List[str], texts: List[List[str]]) -> Dict[str, List[float]]:
        cos_sim_for_ground_truth = []
        cos_sim_for_avg_emb = []

        num_samples = len(true_labels)
        for i in range(0, num_samples, self.batch_size):
            batch_true_labels = true_labels[i : i + self.batch_size]
            batch_generated_labels = generated_labels[i : i + self.batch_size]
            batch_text_groups = texts[i : i + self.batch_size]

            true_label_embeddings = self.encoder.encode_batch(batch_true_labels)
            generated_label_embeddings = self.encoder.encode_batch(batch_generated_labels)
            avg_embeddings = [self.encoder.average_embedding_for_texts(text_group) for text_group in batch_text_groups]

            for true_label_embedding, generated_label_embedding, avg_embedding in zip(true_label_embeddings, generated_label_embeddings, avg_embeddings):
                cos_sim_gt = self.encoder.similarity(true_label_embedding, generated_label_embedding)
                cos_sim_avg = self.encoder.similarity(avg_embedding, generated_label_embedding)

                cos_sim_for_ground_truth.append(cos_sim_gt)
                cos_sim_for_avg_emb.append(cos_sim_avg)

        return {
            "cos_sim_for_ground_truth": cos_sim_for_ground_truth,
            "cos_sim_for_avg_emb": cos_sim_for_avg_emb,
        }

    def calculate_perplexity(self, reference_texts, generated_texts, batch_size=8):
        # Placeholder for perplexity calculation
        pass

    def calculate_rouge_n(self, reference_summaries, generated_summaries, n=1):
        # Placeholder for ROUGE-N calculation
        pass

    def calculate_metrics(self, true_labels, generated_labels, texts, reference_texts, generated_texts, reference_summaries, generated_summaries):
        # Placeholder for overall metrics calculation
        pass
