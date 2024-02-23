
import math
from argparse import ArgumentParser
from tqdm import tqdm
import common_utils


# Hyperparemeters of the search - assigned with commandline arguments
MAX_LEN = None
step_size = None
num_gens = None

# Phrase we're looking for - assigned with commandline arguments
target_phrase = None
target_embedding = None

# Best found state
best_state = None
best_score = -1.5 # score -1.5 as cos similiarity is in range [-1, 1]


"""
    Node class in MCTS search tree.
        state - string that the node represents
        parent - pointer to parent node
        children - pointers to children nodes identified by state
        visits - number of visits to this node
        eval - value in this node
    Attributes visits and eval are essential to MCTS.
"""
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0 # Value for MCTS search
        self.score = float(eval(state)) # Cos similarity for this sentence


"""
    We're selecting a child which scores max on Upper Confidence Bound (UCT) score
"""
def select(node):
    l = list(node.children.values())
    l.append(node)
    return max(l, key=uct_score)


"""
    Calculating UCT score.
    If node is not visited - assign infinity to it, else calculate the score with formula
    node_value / node_visits + sqrt(2 * ln(parent_visits) / node_visits)
"""
def uct_score(node):
    if node.visits == 0:
        return float("inf")
    parent_visits = node.parent.visits if node.parent is not None else 1
    return (node.value / node.visits) + math.sqrt(
        2 * math.log(parent_visits) / node.visits
    )


"""
    Getting random child for our node. 
    We're generating only few most possible sentence continuations, for two reasons:
    1. Reducing search space.
    2. Considering only most humanlike sentence continuations.
"""
def get_random_child(node):
    result = common_utils.generator(
        node.state,  # sentence in node
        max_length=len(node.state) + step_size,  # setting max length
        num_return_sequences=num_gens,
        pad_token_id=50256 # Does nothing - default value for gpt-2
    )
    child_state = common_utils.random.choice(result)["generated_text"] # TODO: should this be weigthed by LLM perplexity?
    child = Node(child_state, parent=node)
    if child_state not in node.children:
        node.children[child_state] = child
    return node.children[child_state]


def expand(node):
    return get_random_child(node)


def simulate(node):
    while not is_terminal(node):
        node = get_random_child(node)
    # Here node is terminal, so we set the value of the game to the same as of cos similarity
    node.value = node.score
    return node.value


def backpropagate(node, value):
    while node is not None:
        node.visits += 1
        node.value += value
        node = node.parent


def is_terminal(node):
    return len(node.state) >= MAX_LEN


def eval(state):
    global best_score, best_state
    embedding = common_utils.embedder.encode(state)
    score = embedding @ target_embedding
    # Always when we eval state we check if this_state > best_state
    if score > best_score:
        best_state = state
        best_score = score
        print("found new best state ", best_state, " with score ", best_score)
    return score


def mcts(root_state, iterations):
    global best_state, best_score
    root_node = Node(root_state) # Create root_node
    best_state = root_node.state
    best_score = root_node.value

    for _ in tqdm(range(iterations), desc="MCTS progress"):
        selected_node = select(root_node)
        if not is_terminal(selected_node):
            child_node = expand(selected_node)
            simulation_result = simulate(child_node)
            backpropagate(child_node, simulation_result)
        else:
            backpropagate(selected_node, eval(selected_node.state))
    return root_node


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="MCTS solution",
        description="Finding human-understandable word minimizing cosine similarity with given embedding with the use of LLM and MCTS search.",
    )
    parser.add_argument(
        "--iter-num", type=int, help="Hyperparameter to MCTS search.", default=1000
    )
    parser.add_argument(
        "--max-len",
        type=int,
        help="Max length in characters for a searched phrase.",
        default=20,
    )
    parser.add_argument(
        "--step-size",
        type=int,
        help="How many characters should be generated at most in one step of MCTS.",
        default=2,
    )
    parser.add_argument(
        "--num-gens",
        type=int,
        help="How many sentence continuations should LLM propose",
        default=10,
    )
    parser.add_argument(
        "--target-phrase", type=str, help="Target phrase to decode.", required=True
    )
    args = parser.parse_args()

    # INITIAL SETUP
    MAX_LEN = args.max_len
    step_size = args.step_size
    num_gens = args.num_gens
    target_phrase = (
        args.target_phrase
    )  # phrase that we want to obtain (unknown to the algorithm)
    target_embedding = common_utils.embedder.encode(
        target_phrase
    )  # embedding of target_phrase (known to the algorithm)

    # START SEARCH
    root_node = mcts("", iterations=args.iter_num)

    # AFTER SEARCH FINISHES
    print("best state: ", best_state, " with similarity ", eval(best_state))

    # def print_tree(node):
    #    print(node.state)
    #    for child in node.children.values():
    #        print_tree(child)
