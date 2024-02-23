from transformers import pipeline, set_seed
from sentence_transformers import SentenceTransformer
import random


### If you're gonna use any embedder / generator make sure that they're initialized in this file
### This is in order not to get confused and use different model for different experiments
### In future if we want to allow for using different models for text generation and embeddings
### we can create here different initializers and eval functions sharing common interface

random.seed(42) # for reproducibility

# OUR LLM + EMBEDDING SETUP
set_seed(42)  # for reproducibility
generator = pipeline("text-generation", model="gpt2")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
