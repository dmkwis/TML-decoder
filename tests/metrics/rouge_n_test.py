import pytest
from tml_decoder.metrics.rouge_n import evaluate_rouge_n  # Adjust the import according to your actual file structure

class TestROUGEN:
    @pytest.mark.parametrize("references, candidates, n, expected", [
        # Test case 1: Simple exact match
        (["The quick brown fox jumps over the lazy dog"], ["The quick brown fox jumps over the lazy dog"], 1, {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}),
        # Test case 2: Partial match with different n-gram sizes
        (["The quick brown fox jumps over the lazy dog"], ["A quick brown dog jumps over the lazy fox"], 2, {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}),
        # Test case 3: No match
        (["The quick brown fox jumps over the lazy dog"], ["Completely unrelated sentence"], 1, {'precision': 0, 'recall': 0, 'f1': 0}),
        # Test case 4: Test with empty strings
        ([""], [""], 1, {'precision': 0, 'recall': 0, 'f1': 0}),
        # Add more test cases as needed
    ])
    def test_evaluate_rouge_n(self, references, candidates, n, expected):
        result = evaluate_rouge_n(references, candidates, n)
        assert result['precision'] == expected['precision'], f"Expected precision {expected['precision']}, got {result['precision']}"
        assert result['recall'] == expected['recall'], f"Expected recall {expected['recall']}, got {result['recall']}"
        assert result['f1'] == expected['f1'], f"Expected f1 {expected['f1']}, got {result['f1']}"
