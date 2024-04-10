import pytest
from tml_decoder.models.mcts_model import MCTSModel, Node


@pytest.fixture
def mock_encoder(mocker):
    return mocker.Mock()


@pytest.fixture
def mock_generator(mocker):
    return mocker.Mock()


@pytest.fixture
def mock_guide(mocker):
    return mocker.Mock()


@pytest.fixture
def mcts_model(mock_encoder, mock_generator, mock_guide):
    initial_prompt = "These documents describe"
    model = MCTSModel(
        encoder=mock_encoder,
        generator=mock_generator,
        guide=mock_guide,
        iter_num=10,
        max_len=100,
        min_result_len=10,
        initial_prompt=initial_prompt,
    )
    return model


class TestMCTSModel:
    def test_initialization(self, mcts_model):
        """Test model initialization with an initial prompt."""
        assert mcts_model.initial_prompt == "These documents describe"

    def test_expand_functionality_real_node(self, mcts_model, mocker):
        """
        Test the expand functionality using the real Node class.
        This test mocks the generator's generate method to return a predefined list of continuations,
        then checks if expanding a node with one of these continuations works as expected.
        """
        # Mock the generator's generate method to return a specific set of states.
        mocked_states = [
            "These documents describe something interesting.",
            "These documents describe something boring.",
        ]
        mocker.patch.object(
            mcts_model.generator, "generate", return_value=mocked_states
        )

        # Assume the guide's choose_next method simply returns the first unexplored state.
        mocker.patch.object(
            mcts_model.guide, "choose_next", side_effect=lambda actions, _: actions[0]
        )

        initial_state = "These documents describe"
        initial_node = Node(
            initial_state,
            mcts_model.encoder,
            mcts_model.generator,
            mcts_model.target_embedding,
            None,
        )

        # Perform the expansion.
        new_child = mcts_model.expand(initial_node)

        # Check that the new child node has the expected state.
        expected_state = mocked_states[0]
        assert (
            new_child.state == expected_state
        ), "The new child node does not have the expected state."

        # Check that the new child node is correctly linked as a child of the initial node.
        assert (
            initial_node.children.get(expected_state) == new_child
        ), "The new child node is not correctly linked to the initial node."
