from typing import List

from numpy import ndarray
from tml_decoder.models.guides.abstract_guide import AbstractGuide
import random

class RandomGuide(AbstractGuide):
    def choose_next(self, propositions: List[str], target_embedding: ndarray) -> str:
        return random.choice(propositions)
    
    def reset(self):
        pass