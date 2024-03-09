from rouge_score import rouge_scorer

def evaluate_rouge_n(reference_summaries, generated_summaries, n=1):
    """
    Evaluate ROUGE-N score for a set of generated summaries against reference summaries.
    
    Parameters:
    - reference_summaries: A list of reference summaries.
    - generated_summaries: A list of generated summaries.
    - n: The n-gram length for ROUGE-N calculation.
    
    Returns:
    - A dictionary containing the average ROUGE-N precision, recall, and F1 score.
    """
    scorer = rouge_scorer.RougeScorer([f'rouge{n}'], use_stemmer=True)
    scores = []

    for reference, generated in zip(reference_summaries, generated_summaries):
        score = scorer.score(reference, generated)
        scores.append(score[f'rouge{n}'])

    # Calculate average scores
    avg_precision = sum(score.precision for score in scores) / len(scores)
    avg_recall = sum(score.recall for score in scores) / len(scores)
    avg_f1 = sum(score.fmeasure for score in scores) / len(scores)

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }
