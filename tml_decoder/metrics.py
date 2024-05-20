from typing import Dict, List

from rouge_score import rouge_scorer

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

    def calculate_perplexity(self, reference_texts: List[str], generated_texts: List[str], batch_size: int = 8) -> Dict[str, List[float]]:
        reference_perplexities = []
        generated_perplexities = []

        num_samples = len(reference_texts)
        for i in range(0, num_samples, batch_size):
            batch_reference_texts = reference_texts[i : i + batch_size]
            batch_generated_texts = generated_texts[i : i + batch_size]

            batch_reference_perplexities = [self.generator.calculate_perplexity(text) for text in batch_reference_texts]
            batch_generated_perplexities = [self.generator.calculate_perplexity(text) for text in batch_generated_texts]

            reference_perplexities.extend(batch_reference_perplexities)
            generated_perplexities.extend(batch_generated_perplexities)

        return {
            "reference_perplexity": reference_perplexities,
            "generated_perplexity": generated_perplexities,
        }

    def evaluate_rouge_n(self, reference_summaries: List[str], generated_summaries: List[str], n: int = 1) -> Dict[str, float]:
        """
        Evaluate ROUGE-N score for a set of generated summaries against reference summaries.

        Parameters:
        - reference_summaries: A list of reference summaries.
        - generated_summaries: A list of generated summaries.
        - n: The n-gram length for ROUGE-N calculation.

        Returns:
        - A dictionary containing the average ROUGE-N precision, recall, and F1 score.
        """
        scorer = rouge_scorer.RougeScorer([f"rouge{n}"], use_stemmer=True)
        scores = []
        for reference, generated in zip(reference_summaries, generated_summaries):
            score = scorer.score(reference, generated)
            scores.append(score[f"rouge{n}"])
        # Calculate average scores
        avg_precision = sum(score.precision for score in scores) / len(scores)
        avg_recall = sum(score.recall for score in scores) / len(scores)
        avg_f1 = sum(score.fmeasure for score in scores) / len(scores)
        return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}

    def calculate_metrics(self, true_labels, generated_labels, texts, reference_texts, generated_texts, reference_summaries, generated_summaries):
        # Placeholder for overall metrics calculation
        pass
