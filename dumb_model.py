from typing import Iterable
from abstractmodel import AbstractLabelModel


class DumbModel(AbstractLabelModel):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self):
        return "dumb model"

    def get_label(self, texts: Iterable[str]) -> str:
        super().get_label(texts)
        return "Terrible label :("
