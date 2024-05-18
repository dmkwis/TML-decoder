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
        # Implement the test for calculate_perplexity when the function is defined
        pass

    def test_calculate_rouge_n(self, metrics):
        # Implement the test for calculate_rouge_n when the function is defined
        pass

    def test_calculate_metrics(self, metrics):
        # Implement the test for calculate_metrics when the function is defined
        pass
