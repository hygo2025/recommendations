import importlib
from collections.abc import Callable

from tqdm import tqdm

from . import defaults

from ..base import RecommenderModel
from ..evaluation import Evaluator

class TrialStopException(Exception):
    """Generic exception raised for errors in trials."""
    def __init__(self):
        self.message = f'Stopping the trial. Reason: {self.__class__.__name__}'
        super().__init__(self.message)

class EarlyStopping(TrialStopException): pass

def trainer(model: RecommenderModel, evaluator: Evaluator) -> None:
    '''
    A standard boilerplate for training a neural network.
    ====================================================

    Supports pruning/reporting based on callbacks. Callback
    must be available in the `evaluator`.
    '''
    max_epochs = model.config['max_epochs']
    validation_interval = defaults.validation_interval
    assert validation_interval <= max_epochs, 'Number of epochs is too small. Won\'t validate.'
    # TODO warn if max_epochs % validation_interval != 0

    pbar = tqdm(range(max_epochs), leave=False)
    for epoch in pbar:
        pbar.set_description(f"Processando época {epoch} de {max_epochs}")
        loss = model.train_epoch()
        if (epoch + 1) % validation_interval == 0:
            try:
                evaluator.submit(model, step=epoch, args=(loss,))
            except EarlyStopping:
                break


def train_data_format(model_name):
    sparse = ['svd', 'random', 'mostpop']
    sequential_packed = ['sasrec', 'sasrecb', 'hypsasrec', 'hypsasrecb']
    sequential = []
    sequential_typed = []
    if model_name in sparse:
        return 'sparse'
    if model_name in sequential:
        return 'sequential' # pandas Series
    if model_name in sequential_packed:
        return 'sequential_packed' # csr-like format
    if model_name in sequential_typed:
        return 'sequential_typed' # numba dict
    return 'default'

def test_data_format(next_item_only):
    if next_item_only: # e.g., lorentzfm is not suitable for sequential prediction
        return ('interactions', dict(stepwise=True, max_steps=1))
    return 'sequential' # 'sequential' will enable vectorized evaluation, if a model supports it