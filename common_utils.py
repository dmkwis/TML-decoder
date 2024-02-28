from typing import Any
from sentence_transformers import SentenceTransformer
import random
from abstract_encoder import AbstractEncoder
from abstract_generator import AbstractGenerator
from abstract_model import AbstractLabelModel

from gpt2_generator import GPT2Generator
from mcts_model import MCTSModel
from miniLM_encoder import MiniLMEncoder


### If you're gonna use any embedder / generator make sure that they're initialized in this file
### This is in order not to get confused and use different model for different experiments
### In future if we want to allow for using different models for text generation and embeddings
### we can create here different initializers and eval functions sharing common interface

random.seed(42)  # for reproducibility

# OUR LLM + EMBEDDING SETUP


def get_generator(name: str, *args: Any, **kwargs: Any) -> AbstractGenerator:
    """
    Returns a generator based on the given name.

    Args:
        name: The name of the generator.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        AbstractGenerator: An instance of the generator.

    Raises:
        NotImplementedError: If the generator with the given name is not implemented.
    """
    if name == "gpt2":
        return GPT2Generator(*args, **kwargs)

    raise NotImplementedError(f"Generator {name} not implemented")


def get_encoder(name: str, *args: Any, **kwargs: Any) -> AbstractEncoder:
    """
    Returns an encoder based on the specified name.

    Args:
        name: The name of the encoder.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        AbstractEncoder: An instance of the encoder.

    Raises:
        NotImplementedError: If the specified encoder is not implemented.
    """
    if name == "MiniLM":
        return MiniLMEncoder(*args, **kwargs)

    raise NotImplementedError(f"Encoder {name} not implemented")


def get_model(name: str, *args: Any, **kwargs: Any) -> AbstractLabelModel:
    """
    Returns an instance of a model based on the given name.

    Args:
        name: The name of the model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        AbstractLabelModel: An instance of the model.

    Raises:
        NotImplementedError: If the model with the given name is not implemented.
    """
    if name == "MCTS":
        return MCTSModel(*args, **kwargs)

    raise NotImplementedError(f"Model {name} not implemented")
