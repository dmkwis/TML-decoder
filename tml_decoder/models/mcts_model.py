import math
import random
from typing import List

from tqdm import tqdm
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.generators.abstract_generator import AbstractGenerator
from tml_decoder.models.abstract_model import AbstractLabelModel
import numpy as np

from tml_decoder.models.guides.abstract_guide import AbstractGuide


class Node:
    def __init__(
        self,
        state,
        encoder: AbstractEncoder,
        generator: AbstractGenerator,
        target_embedding,
        parent=None,
    ):
        self.state = state
        self.parent = parent
        self.all_states = generator.generate(self.state)
        self.children = {}
        self.visits = 0
        self.value = 0  # Value for MCTS search
        self.score = encoder.similarity(
            encoder.encode(self.state), target_embedding
        )  # Cos similarity for this sentence

    def get_unexplored_states(self):
        return list(set(self.all_states).difference(self.children.keys()))

    def is_fully_expanded(self):
        return len(self.get_unexplored_states()) == 0

    def get_all_states(self):
        return self.all_states

    def get_score(self):
        return self.score

    def best_uct(self):
        exploration_weight = 1.41  # Adjust this parameter as needed
        ucts = {
            child.state: child.value / (child.visits + 1e-6)
            + exploration_weight
            * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children.values()
        }
        best_child_state = max(ucts, key=ucts.get)
        return self.children[best_child_state]


class MCTSModel(AbstractLabelModel):
    def __init__(
        self,
        encoder: AbstractEncoder,
        generator: AbstractGenerator,
        guide: AbstractGuide,
        iter_num: int = 100,
        max_len: int = 15,
        min_result_len: int = 3,
        initial_prompt: str = "",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.guide = guide
        self.iter_num = iter_num
        self.max_len = max_len
        self.min_result_len = min_result_len
        self.target_embedding = None
        self.initial_prompt = initial_prompt

    @property
    def name(self):
        return f"MCTS model, encoder: {self.encoder.name}, generator: {self.generator.name}"

    def get_embedding_to_revert(self, texts: List[str]):
        encoded_texts = np.stack([self.encoder.encode(text) for text in texts])
        return self.encoder.average_embedding(encoded_texts)

    def is_terminal_node(self, node):
        return len(node.state) >= self.max_len

    def expand(self, node):
        unexplored_actions = node.get_unexplored_states()
        new_state = self.guide.choose_next(unexplored_actions, self.target_embedding)
        child = Node(
            new_state, self.encoder, self.generator, self.target_embedding, parent=node
        )
        node.children[new_state] = child
        return child

    def select_node(self, node):
        while not self.is_terminal_node(node):
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_uct()
        return node

    def simulate(self, node):
        # Perform a random simulation from the current state and return the result
        while not self.is_terminal_node(node):
            new_state = random.choice(node.get_all_states())
            new_node = Node(
                new_state, self.encoder, self.generator, self.target_embedding, node
            )
            node.children[new_state] = new_node
            node = new_node
        return node.get_score()

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def mcts(self, initial_state, iterations, target_embedding):
        self.guide.reset()
        self.target_embedding = target_embedding
        root_node = Node(
            initial_state, self.encoder, self.generator, self.target_embedding
        )

        for _ in tqdm(range(iterations), "MCTS progress"):
            node = self.select_node(root_node)
            result = self.simulate(node)
            self.backpropagate(node, result)

        def find_best_state(node):
            if len(node.children.values()) == 0:
                return node
            best_from_kids = max(
                map(lambda n: find_best_state(n), node.children.values()),
                key=lambda x: x.score,
            )
            if (
                len(node.state) > self.min_result_len
                and node.score > best_from_kids.score
            ):
                return node
            return best_from_kids

        return find_best_state(root_node).state

    def get_label(self, texts: List[str]) -> str:
        target_embedding = self.get_embedding_to_revert(texts)
        return self.mcts(self.initial_prompt, self.iter_num, target_embedding)
