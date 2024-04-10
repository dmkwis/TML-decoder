from typing import List

from numpy import ndarray
from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.generators.abstract_generator import AbstractGenerator
from tml_decoder.models.guides.abstract_guide import AbstractGuide
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch

from tml_decoder.utils.helper_functions import default_device

class SoftPromptGuide(AbstractGuide):
    def __init__(
        self,
        encoder: AbstractEncoder,
        device=default_device()
    ):
        self.cache_score = {}
        self.encoder = encoder
        self.encoder.freeze_weights()
        self.encoder.unfreeze_embedding_weigths()
        self.criterion = nn.CosineEmbeddingLoss()
        self.unused_token_id = self.encoder.get_token_id(self.encoder.get_unused_token())
        self.cs = nn.CosineSimilarity()
        self.device = device

    def _soft_prompt(self, text: str, target_embedding: ndarray) -> float:
        target_tensor = torch.tensor(target_embedding).to(self.device)
        tokenized_text = self.encoder.tokenize_text(text)
        tokenized_text[-1] = self.unused_token_id
        tokenized_text.append(self.encoder.get_eot_token_id())
        input_ids = torch.tensor(tokenized_text).unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        optimizer = optim.Adam(self.encoder.get_parameters(), lr=1)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
        char_tensor = self.encoder.get_characteristic_tensor_for_token_id(self.unused_token_id)
        best = float("inf")
        best_emb = None
        for _ in range(200):
            optimizer.zero_grad()
            output = self.encoder.raw_encode(input_ids, attention_mask)
            
            loss = self.criterion(output.squeeze(), target_tensor, torch.tensor(1.0).to(self.device))
            loss.backward()
            # set grad of non-new token to 0
            # all ids except new_token_id
            
            self.encoder.zero_grad_for_embeddings(char_tensor)

            optimizer.step()
            scheduler.step()

            if loss.item() < best:
                best = loss.item()
                best_emb = self.encoder.get_embedding_for_token_id(self.unused_token_id).detach().cpu().numpy()
        
        # now discretization
        cossims = []
        for token_id in range(len(self.encoder.get_tokenizer_vocab())):
            token_embedding = self.encoder.get_embedding_for_token_id(token_id).cpu().numpy()
            cossims.append(self.cs(torch.Tensor(token_embedding).unsqueeze(0), torch.Tensor(best_emb).unsqueeze(0)).item())
        return max(cossims)

    def _get_score(self, text: str, target_embedding: ndarray) -> float:
        if text not in self.cache_score:
            self.cache_score[text] = self._soft_prompt(text, target_embedding)
        return self.cache_score[text]

    def choose_next(self, propositions: List[str], target_embedding: ndarray) -> str:
        return max(propositions, key=lambda text: self._get_score(text, target_embedding))
