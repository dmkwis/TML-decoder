import torch

from typing import List

from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.models.abstract_model import AbstractLabelModel
from vec2text import invert_embeddings, load_pretrained_corrector
import vec2text
from transformers import AutoTokenizer, AutoModel

from tml_decoder.utils.common_utils import default_device

class Vec2TextModel(AbstractLabelModel):
    def __init__(self, encoder: AbstractEncoder, num_steps=10, device=default_device()):
        assert encoder.name == "gtr-base", "Vec2Text supports only gtr-base encoder"

        self.corrector = load_pretrained_corrector("gtr-base")
        self.encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(device)
        self.num_steps = int(num_steps)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

    @property
    def name(self):
        return "vec2text model"
    
    def get_gtr_embeddings(self, text_list: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(text_list,
                            return_tensors="pt",
                            max_length=128,
                            truncation=True,
                            padding="max_length",).to(self.device)

        with torch.no_grad():
            model_output = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            hidden_state = model_output.last_hidden_state
            embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

        return embeddings

    def get_label(self, texts: List[str]) -> str:
        embeddings = self.get_gtr_embeddings(texts)
        mean_embedding = embeddings.mean(dim=0, keepdim=True).to(self.device)

        [result] = invert_embeddings(
            mean_embedding, self.corrector, num_steps=self.num_steps
        )

        return result
