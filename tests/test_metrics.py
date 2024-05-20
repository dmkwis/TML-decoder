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
        assert len(result["cos_sim_for_ground_truth"]) == len(true_labels)
        assert len(result["cos_sim_for_avg_emb"]) == len(true_labels)
        assert all(isinstance(x, float) for x in result["cos_sim_for_ground_truth"])
        assert all(isinstance(x, float) for x in result["cos_sim_for_avg_emb"])

    def test_calculate_perplexity(self, metrics):
        reference_texts = ["This is a test.", "Another test sentence."]
        generated_texts = ["This is a generated sentence.", "Another generated text."]

        result = metrics.calculate_perplexity(reference_texts, generated_texts)

        assert "reference_perplexity" in result
        assert "generated_perplexity" in result
        assert all(isinstance(x, float) for x in result["reference_perplexity"])
        assert all(isinstance(x, float) for x in result["generated_perplexity"])

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
    def test_evaluate_rouge_n(self, metrics, references, candidates, n, expected):
        result = metrics.evaluate_rouge_n(references, candidates, n)
        assert result["precision"] == expected["precision"], f"Expected precision {expected['precision']}, got {result['precision']}"
        assert result["recall"] == expected["recall"], f"Expected recall {expected['recall']}, got {result['recall']}"
        assert result["f1"] == expected["f1"], f"Expected f1 {expected['f1']}, got {result['f1']}"

    def test_calculate_metrics(self, metrics):
        # Implement the test for calculate_metrics when the function is defined
        pass
