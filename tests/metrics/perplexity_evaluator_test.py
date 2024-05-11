import pytest

from tml_decoder.metrics.perplexity_evaluator import GPT2PerplexityEvaluator


class TestPerplexityEvaluator:
    @pytest.fixture(scope="class")
    def evaluator(self):
        # Setup the evaluator object
        evaluator = GPT2PerplexityEvaluator()
        return evaluator

    def test_calculate_perplexity_short_text(self, evaluator):
        # Test calculate_perplexity with a short text
        text = "The quick brown fox jumps over the lazy dog."
        expected_perplexity_range = (1, 200)  # Hypothetical expected range for demonstration
        perplexity = evaluator.calculate_perplexity([text])[0]
        assert expected_perplexity_range[0] < perplexity < expected_perplexity_range[1], f"Perplexity {perplexity} not within expected range {expected_perplexity_range}"

    def test_calculate_perplexity_empty_text(self, evaluator):
        # Test calculate_perplexity with an empty string
        text = ""
        expected_perplexity = 0  # Assuming a specific behavior for empty texts
        perplexity = evaluator.calculate_perplexity([text])[0]
        assert perplexity == expected_perplexity, f"Expected perplexity for empty text is {expected_perplexity}, got {perplexity}"
