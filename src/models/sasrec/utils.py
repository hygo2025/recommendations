import json
import os
from typing import Union, Optional

import numpy as np
import pandas as pd
import torch
from numba import types
from numba.typed import Dict
from scipy.sparse import csr_matrix

from src.models.sasrec import defaults

def topidx(arr, topn):
    parted = np.argpartition(arr, -topn)[-topn:]
    return parted[np.argsort(-arr[parted])]

def reindex(raw_data, index, filter_invalid=True, names=None):
    '''
    Factorizes column values based on provided pandas index. Allows resetting
    index names. Optionally drops rows with entries not present in the index.
    '''
    if isinstance(index, pd.Index):
        index = [index]

    if isinstance(names, str):
        names = [names]

    if isinstance(names, (list, tuple, pd.Index)):
        for i, name in enumerate(names):
            index[i].name = name

    new_data = raw_data.assign(**{
        idx.name: idx.get_indexer(raw_data[idx.name]) for idx in index
    })

    if filter_invalid:
        # pandas returns -1 if label is not present in the index
        # checking if -1 is present anywhere in data
        maybe_invalid = new_data.eval(
            ' or '.join([f'{idx.name} == -1' for idx in index])
        )
        if maybe_invalid.any():
            print(f'Filtered {maybe_invalid.sum()} invalid observations.')
            new_data = new_data.loc[~maybe_invalid]

    return new_data



def save_config(config: dict, experiment_name: str) -> None:
    """
    Salva a configuração do experimento em um arquivo.

    Parameters
    ----------
    config : dict
        Configuração a ser salva.
    experiment_name : str
        Nome do experimento para identificar o arquivo de configuração.

    Returns
    -------
    None
    """
    filename = os.path.join(defaults.data_dir, f'results/{experiment_name}_config.txt')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)


def fix_torch_seed(seed, conv_determinism=True):
    '''
    Notes:
    -----
    It doesn't fix the CrossEntropy loss non-determinism, to check it set `torch.use_deterministic_algorithms(True)`.
    For more details, see
    https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180

    The `conv_determinism` settings may affect computational performance, see
    https://pytorch.org/docs/stable/notes/randomness.html:

    Also note that it doesn't fix possible non-determinism in loss calculation, see:
    https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/8

    For debugging use torch.use_deterministic_algorithms(True)
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if conv_determinism:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_torch_device(device_name=None):
    print(torch.__version__)
    print(torch.version.cuda)
    if device_name is None:
        device_name = 'cpu'
        if torch.cuda.is_available():
            device_name = f'cuda:{torch.cuda.current_device()}'
    device = torch.device(device_name)
    return device


def dump_intermediate_results(results: dict, filename: str, indent: int=4):
    with open(filename, 'a+') as f:
        if os.stat(filename).st_size == 0:
            f.write('[')
        else:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 1, os.SEEK_SET)
            f.truncate()
            f.write(', ')
        json.dump(results, f, indent=indent)
        f.write(']')

def dump_trial_results(results: dict, config: dict, experiment_name: str, trial_id: Optional[int] = None) -> None:
    """
    Salva os resultados do experimento e a configuração associada em um arquivo.

    Parameters
    ----------
    results : dict
        Dicionário contendo os resultados do experimento. Pode conter as chaves 'history' e 'best'
        (para aprendizado iterativo) ou 'metrics' para aprendizado não iterativo.
    config : dict
        Configuração utilizada no experimento.
    experiment_name : str
        Nome do experimento, usado para identificar o arquivo de resultados.
    trial_id : Optional[int], default: None
        Identificador opcional do experimento.
    """
    filename = os.path.join(defaults.data_dir, f'results/{experiment_name}_results.txt')

    # Define o dicionário de resultados conforme as chaves disponíveis
    if 'history' in results:
        results_dict = {'results': results['history']}
        if 'best' in results:
            results_dict['best'] = results['best']
    else:
        results_dict = {'results': results.get('metrics', results)}

    results_dict['config'] = config
    results_dict['trial_id'] = trial_id

    dump_intermediate_results(results_dict, filename)