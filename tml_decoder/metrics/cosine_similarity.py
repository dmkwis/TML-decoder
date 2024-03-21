from typing import List, Dict

from tml_decoder.encoders.abstract_encoder import AbstractEncoder

def evaluate_cosine_similarity(encoder: AbstractEncoder, true_labels: List[str], generated_labels: List[str], texts: List[List[str]]) -> Dict[str, List[float]]:
    cos_sim_for_ground_truth = []
    cos_sim_for_avg_emb = []
    for true_label, generated_label, text_group in zip(true_labels, generated_labels, texts):
        true_label_embedding = encoder.encode(true_label)
        generated_label_embedding = encoder.encode(generated_label)
        avg_embedding = encoder.average_embedding_for_texts(text_group)

        cos_sim_gt = encoder.similarity(true_label_embedding, generated_label_embedding)
        cos_sim_avg = encoder.similarity(avg_embedding, generated_label_embedding)

        cos_sim_for_ground_truth.append(cos_sim_gt)
        cos_sim_for_avg_emb.append(cos_sim_avg)
    
    return {
        "cos_sim_for_ground_truth": cos_sim_for_ground_truth,
        "cos_sim_for_avg_emb": cos_sim_for_avg_emb,
    }
