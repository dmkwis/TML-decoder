import random
import math
from transformers import pipeline
from sentence_transformers import SentenceTransformer


### OUR SETUP

generator = pipeline("text-generation", model="gpt2")

target_sentence = "bear attack"

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
target_embedding = embedder.encode(target_sentence)

MAX_LEN = 20
best_state = None
best_score = -1.5

num_gens = 20
step_size = 2

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = eval(state)


def select(node):
    l = list(node.children.values())
    l.append(node)
    return max(l, key=uct_score)




def uct_score(node):
    if node.visits == 0:
        return float("inf")
    z = node.parent.visits if node.parent is not None else 1
    return (node.value / node.visits) + math.sqrt(
        2 * z / node.visits
    )


def expand(node):
    return get_random_child(node)


def simulate(node):
    while not is_terminal(node):
        node = get_random_child(node)
    return node.value


def backpropagate(node, value):
    while node is not None:
        node.visits += 1
        node.value = max(node.value, value)
        node = node.parent


def get_random_child(node):
    result = generator(
        node.state, max_length=len(node.state) + step_size, num_return_sequences=num_gens
    )
    child_state = random.choice(result)["generated_text"]
    child = Node(child_state, parent=node)
    if child_state not in node.children:
        node.children[child_state] = child
    return node.children[child_state]


def is_terminal(node):
    return len(node.state) >= MAX_LEN


def eval(state):
    global best_state, best_score
    embedding = embedder.encode(state)
    score = embedding @ target_embedding
    if score > best_score:
        best_state = state
        best_score = score
        print("found new best state ", best_state, " with score ", best_score)
    return score


def mcts(root_state, iterations):
    global best_state
    root_node = Node(root_state)
    best_state = root_state

    for iter in range(iterations):
        print(f"iteration {iter}")
        selected_node = select(root_node)
        if not is_terminal(selected_node):
            child_node = expand(selected_node)
            simulation_result = simulate(child_node)
            backpropagate(child_node, simulation_result)
        else:
            backpropagate(selected_node, eval(selected_node.state))
    return root_node


# Example usage:
initial_state = ""
root_node = mcts(initial_state, iterations=100)
print("best state: ", best_state, " with similarity ", eval(best_state))
def print_tree(node):
    print(node.state)
    for child in node.children.values():
        print_tree(child)
