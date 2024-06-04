from typing import Dict, List, Optional

from rouge_score import rouge_scorer
from tqdm import tqdm

from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.generators.abstract_generator import AbstractGenerator


class Metrics:
    def __init__(self, encoder: AbstractEncoder, generator: AbstractGenerator, batch_size: int = 8, metrics_to_skip: Optional[List[str]] = None):
        self.encoder = encoder
        self.generator = generator
        self.batch_size = batch_size
        self.metrics_to_skip = metrics_to_skip if metrics_to_skip is not None else []

    def calculate_cosine_similarity(self, true_labels: List[str], generated_labels: List[str], texts: List[List[str]]) -> Dict[str, float]:
        if "cosine_similarity" in self.metrics_to_skip:
            return {}

        cos_sim_for_ground_truth = []
        cos_sim_for_avg_emb = []

        num_samples = len(true_labels)
        for i in tqdm(range(0, num_samples, self.batch_size), desc="Calculating cosine similarity"):
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

        avg_cos_sim_gt = sum(cos_sim_for_ground_truth) / len(cos_sim_for_ground_truth)
        avg_cos_sim_avg_emb = sum(cos_sim_for_avg_emb) / len(cos_sim_for_avg_emb)

        return {
            "cos_sim_for_ground_truth": avg_cos_sim_gt,
            "cos_sim_for_avg_emb": avg_cos_sim_avg_emb,
        }

    def calculate_perplexity(self, reference_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
        if "perplexity" in self.metrics_to_skip:
            return {}
        reference_perplexities = []
        generated_perplexities = []

        num_samples = len(reference_texts)
        for i in tqdm(range(0, num_samples, self.batch_size), desc="Calculating perplexity"):
            batch_reference_texts = reference_texts[i : i + self.batch_size]
            batch_generated_texts = generated_texts[i : i + self.batch_size]

            batch_reference_perplexities = self.generator.calculate_perplexity(batch_reference_texts, batch_size=self.batch_size)
            batch_generated_perplexities = self.generator.calculate_perplexity(batch_generated_texts, batch_size=self.batch_size)

            reference_perplexities.extend(batch_reference_perplexities)
            generated_perplexities.extend(batch_generated_perplexities)

        avg_reference_perplexity = sum(reference_perplexities) / len(reference_perplexities)
        avg_generated_perplexity = sum(generated_perplexities) / len(generated_perplexities)

        return {
            "avg_reference_perplexity": avg_reference_perplexity,
            "avg_generated_perplexity": avg_generated_perplexity,
        }

    def calculate_rouge_n(self, reference_summaries: List[str], generated_summaries: List[str], n: int = 1) -> Dict[str, float]:
        """
        Evaluate ROUGE-N score for a set of generated summaries against reference summaries.

        Parameters:
        - reference_summaries: A list of reference summaries.
        - generated_summaries: A list of generated summaries.
        - n: The n-gram length for ROUGE-N calculation.

        Returns:
        - A dictionary containing the average ROUGE-N precision, recall, and F1 score.
        """
        if "rouge_n" in self.metrics_to_skip:
            return {}
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
        metrics_result = {}

        if "cosine_similarity" not in self.metrics_to_skip:
            metrics_result["cosine_similarity"] = self.calculate_cosine_similarity(true_labels, generated_labels, texts)

        if "perplexity" not in self.metrics_to_skip:
            metrics_result["perplexity"] = self.calculate_perplexity(reference_texts, generated_texts)

        if "rouge_n" not in self.metrics_to_skip:
            metrics_result["rouge_n"] = self.calculate_rouge_n(reference_summaries, generated_summaries)
        return metrics_result
