from typing import List

from numpy import ndarray
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from tml_decoder.encoders.abstract_encoder import AbstractEncoder
from tml_decoder.utils.helper_functions import default_device


def soft_prompt(
    encoder: AbstractEncoder,
    text: str,
    target_embedding: ndarray,
    number_of_iter: int = 50,
    number_of_propositions: int = 5,
    device: torch.device = default_device(),
) -> List[str]:
    """
    Generates soft prompts for a given text and target embedding.

    Args:
        encoder (AbstractEncoder): The encoder to be used.
        text (str): The input base text e.g. "Alice has a".
        target_embedding (ndarray): The target embedding.
        number_of_iter (int, optional): The number of iterations for generating soft prompts. Defaults to 50.
        number_of_propositions (int, optional): The number of soft prompts to generate. Defaults to 5.
        device (torch.device, optional): The computation device. Defaults to the system's default device.

    Returns:
        List[str]: A list of generated new similar texts e.g. ["Alice has a cat", "Alice has a dog", ...].
    """
    encoder.freeze_weights()
    encoder.unfreeze_embedding_weights()
    criterion = nn.CosineEmbeddingLoss()
    unused_token_id = encoder.get_token_id(encoder.get_unused_token())
    cs = nn.CosineSimilarity()

    target_tensor = torch.tensor(target_embedding, dtype=torch.float32).to(device)
    tokenized_text = encoder.tokenize_text(text)
    tokenized_text[-1] = unused_token_id
    tokenized_text.append(encoder.get_eot_token_id())

    input_ids = torch.tensor(tokenized_text).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    optimizer = optim.Adam(encoder.get_parameters(), lr=1)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)

    encoder.train()
    best = float("inf")
    best_emb = None
    for _ in range(number_of_iter):
        optimizer.zero_grad()
        output = encoder.raw_encode(input_ids, attention_mask)

        loss = criterion(output.squeeze(), target_tensor.squeeze(), torch.tensor(1.0).to(device))
        loss.backward()

        char_tensor = encoder.get_characteristic_tensor_for_token_id(unused_token_id)
        encoder.zero_grad_for_embeddings(char_tensor)

        optimizer.step()
        scheduler.step()

        if loss.item() < best:
            best = loss.item()
            best_emb = encoder.get_embedding_for_token_id(unused_token_id).detach().cpu().numpy()

    encoder.eval()
    cossims = []
    for token_id in range(len(encoder.get_tokenizer_vocab())):
        token_embedding = encoder.get_embedding_for_token_id(token_id).detach().cpu().numpy()
        cossims.append(
            cs(
                torch.Tensor(token_embedding).unsqueeze(0),
                torch.Tensor(best_emb).unsqueeze(0),
            ).item()
        )

    best_tokens_indices = np.argsort(cossims)[-number_of_propositions:][::-1]
    best_tokens = [encoder.decode([idx]) for idx in best_tokens_indices]

    return best_tokens
