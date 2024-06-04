import pytest

from tml_decoder.encoders.mock_encoder import MockEncoder
from tml_decoder.generators.mock_generator import MockGenerator
from tml_decoder.metrics import Metrics


class TestMetrics:
    @pytest.fixture
    def mock_encoder(self):
        return MockEncoder()

    @pytest.fixture
    def mock_generator(self):
        return MockGenerator()

    @pytest.fixture
    def metrics(self, mock_encoder, mock_generator):
        return Metrics(encoder=mock_encoder, generator=mock_generator, batch_size=2)

    def test_calculate_cosine_similarity(self, metrics):
        true_labels = ["true1", "true2", "true3"]
        generated_labels = ["gen1", "gen2", "gen3"]
        texts = [["text1a", "text1b"], ["text2a", "text2b"], ["text3a", "text3b"]]

        result = metrics.calculate_cosine_similarity(true_labels, generated_labels, texts)

        assert "cos_sim_for_ground_truth" in result
        assert "cos_sim_for_avg_emb" in result
        assert isinstance(result["cos_sim_for_ground_truth"], float)
        assert isinstance(result["cos_sim_for_avg_emb"], float)

    def test_calculate_perplexity(self, metrics):
        reference_texts = ["This is a test.", "Another test sentence."]
        generated_texts = ["This is a generated sentence.", "Another generated text."]

        result = metrics.calculate_perplexity(reference_texts, generated_texts)

        assert "avg_reference_perplexity" in result
        assert "avg_generated_perplexity" in result
        assert isinstance(result["avg_reference_perplexity"], float)
        assert isinstance(result["avg_generated_perplexity"], float)

    @pytest.mark.parametrize(
        "references, candidates, n, expected",
        [
            # Test case 1: Simple exact match
            (
                ["The quick brown fox jumps over the lazy dog"],
                ["The quick brown fox jumps over the lazy dog"],
                1,
                {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            ),
            # Test case 2: Partial match with different n-gram sizes
            (
                ["The quick brown fox jumps over the lazy dog"],
                ["A quick brown dog jumps over the lazy fox"],
                2,
                {"precision": 0.5, "recall": 0.5, "f1": 0.5},
            ),
            # Test case 3: No match
            (
                ["The quick brown fox jumps over the lazy dog"],
                ["Completely unrelated sentence"],
                1,
                {"precision": 0, "recall": 0, "f1": 0},
            ),
            # Test case 4: Test with empty strings
            ([""], [""], 1, {"precision": 0, "recall": 0, "f1": 0}),
            # Add more test cases as needed
        ],
    )
    def test_calculate_rouge_n(self, metrics, references, candidates, n, expected):
        result = metrics.calculate_rouge_n(references, candidates, n)
        assert result["precision"] == expected["precision"], f"Expected precision {expected['precision']}, got {result['precision']}"
        assert result["recall"] == expected["recall"], f"Expected recall {expected['recall']}, got {result['recall']}"
        assert result["f1"] == expected["f1"], f"Expected f1 {expected['f1']}, got {result['f1']}"

    def test_calculate_metrics(self, metrics):
        true_labels = ["true1", "true2"]
        generated_labels = ["gen1", "gen2"]
        texts = [["text1a", "text1b"], ["text2a", "text2b"]]
        reference_texts = ["This is a test.", "Another test sentence."]
        generated_texts = ["This is a generated sentence.", "Another generated text."]
        reference_summaries = ["The quick brown fox jumps over the lazy dog"]
        generated_summaries = ["The quick brown fox jumps over the lazy dog"]

        result = metrics.calculate_metrics(true_labels, generated_labels, texts, reference_texts, generated_texts, reference_summaries, generated_summaries)

        assert "cosine_similarity" in result
        assert "perplexity" in result
        assert "rouge_n" in result
        assert all(isinstance(v, dict) for v in result.values())
