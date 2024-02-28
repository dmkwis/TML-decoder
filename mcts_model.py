import math
import random
from typing import Iterable

from tqdm import tqdm
from abstract_encoder import AbstractEncoder
from abstract_generator import AbstractGenerator
from abstract_model import AbstractLabelModel


class Node:
    def __init__(self, state, encoder: AbstractEncoder, target_embedding, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0  # Value for MCTS search
        self.score = encoder.similarity(
            encoder.encode(self.state), target_embedding
        )  # Cos similarity for this sentence


def uct_score(node):
    if node.visits == 0:
        return float("inf")
    parent_visits = node.parent.visits if node.parent is not None else 1
    return (node.value / node.visits) + math.sqrt(
        2 * math.log(parent_visits) / node.visits
    )


def select(node):
    l = list(node.children.values())
    l.append(node)
    return max(l, key=uct_score)


class MCTSModel(AbstractLabelModel):
    def __init__(
        self,
        encoder: AbstractEncoder,
        generator: AbstractGenerator,
        iter_num=100,
        max_len=20,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.iter_num = iter_num
        self.max_len = max_len
        self.best_score = None
        self.best_state = None
        self.target_embedding = None

    @property
    def name(self):
        return f"MCTS model, encoder: {self.encoder.name}, generator: {self.generator.name}"

    def get_embedding_to_revert(self, texts: Iterable[str]):
        encoded_texts = [self.encoder.encode(text) for text in texts]
        return self.encoder.average_embedding(encoded_texts)

    def is_terminal(self, node):
        return len(node.state) >= self.max_len

    def get_best_result(self, n: Node):
        if n.score >= self.best_score:
            self.best_score = n.score
            self.best_state = n.state

    def get_random_child(self, node):
        results = self.generator.generate(node.state)
        child_state = random.choice(
            results
        )  # TODO: should this be weighted by LLM perplexity?
        child = Node(child_state, self.encoder, self.target_embedding, parent=node)
        self.get_best_result(child)
        if child_state not in node.children:
            node.children[child_state] = child
        return node.children[child_state]

    def expand(self, node):
        return self.get_random_child(node)

    def simulate(self, node):
        while not self.is_terminal(node):
            node = self.get_random_child(node)
        # Here node is terminal, so we set the value of the game to the same as of cos similarity
        node.value = node.score
        return node.value

    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def mcts(self, initial_state, iterations, target_embedding):
        self.target_embedding = target_embedding
        root_node = Node(
            initial_state, self.encoder, self.target_embedding
        )  # Create root_node
        self.best_state = root_node.state
        self.best_score = root_node.value

        for _ in tqdm(range(iterations), desc="MCTS progress"):
            selected_node = select(root_node)
            if not self.is_terminal(selected_node):
                child_node = self.expand(selected_node)
                simulation_result = self.simulate(child_node)
                self.backpropagate(child_node, simulation_result)
            else:
                self.backpropagate(selected_node, eval(selected_node.state))
        return self.best_state

    def get_label(self, texts: Iterable[str]) -> str:
        target_embedding = self.get_embedding_to_revert(texts)
        return self.mcts("", self.iter_num, target_embedding)
