from typing import Any
import random
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.generators.abstract_generator import AbstractGenerator
from tml_decoder.models.abstract_model import AbstractLabelModel
from tml_decoder.models.beam_search_model import BeamSearchModel
from tml_decoder.models.dumb_model import DumbModel

from tml_decoder.generators.gpt2_generator import GPT2Generator
from tml_decoder.models.mcts_model import MCTSModel
from tml_decoder.encoders.mini_lm_encoder import MiniLMEncoder


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
    if name == "dumb":
        return DumbModel()
    if name == "MCTS":
        generator = get_generator("gpt2")
        return MCTSModel(generator=generator, *args, **kwargs)
    if name == "beam":
        generator = get_generator("gpt2")
        return BeamSearchModel(generator=generator, *args, **kwargs)

    raise NotImplementedError(f"Model {name} not implemented")
