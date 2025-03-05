import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


def get_positive2negatives(num_items: int, num_samples: int = 100) -> Dict[int, List[int]]:
    """
    Cria um dicionário que mapeia cada amostra positiva (inteiro) para uma lista de amostras negativas (inteiros).

    Para cada amostra positiva (valor de 1 até num_items), os candidatos negativos são todos os inteiros
    exceto o próprio valor positivo. Em seguida, uma amostra aleatória de tamanho `num_samples` é selecionada
    sem reposição a partir desses candidatos.

    Args:
        num_items (int): Número total de itens.
        num_samples (int, optional): Número de amostras negativas a serem selecionadas para cada amostra positiva.
                                     Padrão é 100.

    Returns:
        Dict[int, List[int]]: Dicionário onde cada chave é uma amostra positiva e o valor associado é a lista de
                              amostras negativas.
    """
    all_samples = np.arange(1, num_items + 1)
    positive2negatives = {
        positive_sample: np.random.choice(
            np.r_[np.arange(positive_sample), np.arange(positive_sample + 1, num_items + 1)],
            size=num_samples,
            replace=False
        ).tolist()
        for positive_sample in tqdm(all_samples, desc="Creating positive2negatives", total=len(all_samples))
    }
    return positive2negatives


def get_negative_samples(
        positive2negatives: Dict[int, List[int]],
        positive_seqs: torch.Tensor,
        num_samples: int = 1,
) -> torch.Tensor:
    """
    Gera um tensor com amostras negativas a partir das sequências positivas.

    Para cada elemento em 'positive_seqs' que seja diferente de zero,
    seleciona aleatoriamente uma ou mais amostras negativas do dicionário 'positive2negatives'.
    Caso o valor seja zero, a posição é ignorada (permanece zero).

    Args:
        positive2negatives (Dict[int, List[int]]): Dicionário que mapeia uma amostra positiva para uma lista de amostras negativas.
        positive_seqs (torch.Tensor): Tensor contendo as sequências positivas.
        num_samples (int, optional): Número de amostras negativas a serem selecionadas para cada posição (padrão é 1).
                                     Apenas a primeira amostra selecionada é utilizada.

    Returns:
        torch.Tensor: Tensor com o mesmo shape de 'positive_seqs', onde cada valor positivo foi substituído
                      por uma amostra negativa selecionada aleatoriamente.
    """
    negative_seqs = torch.zeros_like(positive_seqs, dtype=torch.long)

    for row_idx, row in enumerate(positive_seqs):
        for col_idx, sample in enumerate(row):
            positive_sample = sample.item()
            if positive_sample == 0:
                continue

            # Seleciona amostras negativas para a amostra positiva
            negative_samples = positive2negatives[positive_sample]
            selected_negative = np.random.choice(negative_samples, size=num_samples, replace=False)

            # Atribui apenas o primeiro sample negativo selecionado
            negative_seqs[row_idx, col_idx] = selected_negative[0]

    return negative_seqs

def get_torch_device(device_name=None):
    if device_name is None:
        device_name = 'cpu'
        if torch.cuda.is_available():
            device_name = f'cuda:{torch.cuda.current_device()}'
    device = torch.device(device_name)
    return device


def get_output_name(
        data_filename: str,
        lr: float,
        batch_size: int,
        early_stop_epoch: int,
        num_epochs: int,
        random_seed: int,
        timestamp: str
) -> str:
    data_name, _ = os.path.splitext(data_filename)

    output_name = (
        f"sasrec-{data_name}_"
        f"lr-{lr}_"
        f"batch-size-{batch_size}_"
        f"early-stop-{early_stop_epoch}"
        f"num-epochs-{num_epochs}_"
        f"seed-{random_seed}_"
        f"{timestamp}"
    )

    return output_name