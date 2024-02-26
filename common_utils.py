from sentence_transformers import SentenceTransformer
import random
from abstract_encoder import AbstractEncoder
from abstract_generator import AbstractGenerator

from gpt2_generator import GPT2Generator
from miniLM_encoder import MiniLMEncoder



### If you're gonna use any embedder / generator make sure that they're initialized in this file
### This is in order not to get confused and use different model for different experiments
### In future if we want to allow for using different models for text generation and embeddings
### we can create here different initializers and eval functions sharing common interface

random.seed(42) # for reproducibility

# OUR LLM + EMBEDDING SETUP

def get_generator(name: str, *args, **kwargs) -> AbstractGenerator:
    if name == "gpt2":
        return GPT2Generator(*args, **kwargs)
    return None

def get_encoder(name: str, *args, **kwargs) -> AbstractEncoder:
    if name == "MiniLM":
        return MiniLMEncoder(*args, **kwargs)
    return None