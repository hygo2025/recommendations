import logging
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from src.models.evaluation import Evaluator
from src.models.sasrec import defaults
from src.models.sasrec.data.movielens_data_setup import MovieLensDataSetup
from src.models.sasrec.data.processor import DataSet
from src.models.sasrec.m_model.sampler import packed_sequence_batch_sampler
from src.models.sasrec.m_model.sasrec_recommender import SASRecModel
from src.models.sasrec.utils import save_config, dump_trial_results, fix_torch_seed
from src.runners.abstract_runnner import AbstractRunner
from src.utils.enums import MovieLensDataset, TargetMetric
from src.utils.logger import Logger


class TrialStopException(Exception):
    """Generic exception raised for errors in trials."""
    def __init__(self):
        self.message = f'Stopping the trial. Reason: {self.__class__.__name__}'
        super().__init__(self.message)

class EarlyStopping(TrialStopException): pass



class SasRecRunner(AbstractRunner):
    def __init__(self,
                 model: str = 'sasrec',
                 dataset: MovieLensDataset = MovieLensDataset.ML_1M,
                 time_offset: float = 0.95,
                 target_metric: TargetMetric = TargetMetric.NDCG,
                 topn: int = 10,
                 config_path: Optional[str] = None,
                 exhaustive: bool = False,
                 grid_steps: Optional[int] = None,  # 0 significa execução infinita; None usará o valor padrão
                 check_best: bool = True,
                 save_config: bool = True,
                 dump_results: bool = False,
                 es_tol: float = 0.001,
                 es_max_steps: int = 2,
                 next_item_only: bool = False,
                 study_name: Optional[str] = None,
                 storage: str = 'redis'  # Valores permitidos: 'sqlite' ou 'redis'
                 ):
        self.model = model
        self.dataset = dataset
        self.time_offset = time_offset
        self.target_metric = target_metric
        self.topn = topn
        self.config_path = config_path
        self.exhaustive = exhaustive
        self.grid_steps = grid_steps
        self.check_best = check_best
        self.save_config = save_config
        self.dump_results = dump_results
        self.es_tol = es_tol
        self.es_max_steps = es_max_steps
        self.next_item_only = next_item_only
        self.study_name = study_name
        self.storage = storage
        self.logger = Logger.get_logger("SasRecRunner")

    def run_experiment(self, datapack, config):
        # Cria a instância do dataset e inicializa os formatos necessários
        dataset = DataSet(
            datapack,
            name=self.dataset,
            train_format='sequential_packed',
            test_format='sequential'
        )
        dataset.initialize_formats({'train': 'sequential'})
        dataset.info()

        # Prepara a rotina de treinamento do modelo
        evaluator = Evaluator(dataset, self.topn)

        self.logger.info(f'Starting training with configuration: {config}')
        self.logger.info(f'Target metric: {self.target_metric.value.upper()}@{self.topn}')

        # Executa o treinamento e avaliação
        dataset = evaluator.dataset
        n_items = len(dataset.item_index)
        fix_torch_seed(config.get('seed', None))
        model = SASRecModel(config, n_items)


        indices, sizes = dataset.train
        sampler = packed_sequence_batch_sampler(
            indices, sizes, n_items,
            batch_size=config['batch_size'],
            maxlen=config['maxlen'],
            seed=config['sampler_seed'],
        )
        n_batches = (len(sizes) - 1) // config['batch_size']

        max_epochs = config['max_epochs']
        validation_interval = defaults.validation_interval
        assert validation_interval <= max_epochs, 'Number of epochs is too small. Won\'t validate.'
        # TODO warn if max_epochs % validation_interval != 0

        pbar = tqdm(range(max_epochs), leave=False )

        for epoch in pbar:
            pbar.set_description(f"Processando época {epoch} de {max_epochs}")
            loss = model.train_epoch(sampler, n_batches)
            if (epoch + 1) % validation_interval == 0:
                try:
                    evaluator.submit(model, step=epoch, args=(loss,))
                except EarlyStopping:
                    break


        # Recupera o score a partir dos resultados da avaliação
        try:
            best_results = evaluator.results['best']
        except KeyError:
            results = evaluator.most_recent_results
        else:
            results = evaluator.results[best_results['step']]
        score = results.loc[f'{self.target_metric.value.upper()}@{self.topn}', 'score']

        return score, results, evaluator


    def run(self) -> None:
        ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        study_name = f'{self.model}_{self.dataset}_{self.target_metric}_{ts}'
        tune_datapack, test_datapack = MovieLensDataSetup(self.dataset, time_offset_q=[self.time_offset] * 2).prepare_data()

        config = return_config()
        if config.get('maxlen') is None:
            config['maxlen'] = defaults.sequence_length_movies

        if self.save_config:
            save_config(config=config, experiment_name=study_name)

        score, results, evaluator = self.run_experiment(tune_datapack,  config)
        self.logger.info(f"Training score: {score}")

        # Caso a flag de teste esteja ativa, executa a avaliação no conjunto de teste
        if self.check_best:
            self.logger.info("Running test evaluation on best configuration...")
            test_score, test_results, test_evaluator = self.run_experiment(test_datapack, config)
            self.logger.info(f"Test results for the provided parameters:\n{test_results}")
            self.logger.info(test_evaluator.results)
            if self.dump_results:
                dump_trial_results(test_evaluator.results, config, f'{study_name}_TEST')


def test_data_format(next_item_only):
    if next_item_only: # e.g., lorentzfm is not suitable for sequential prediction
        return ('interactions', dict(stepwise=True, max_steps=1))
    return 'sequential' # 'sequential' will enable v

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

def return_config() -> dict:
    return {'batch_size': 256, 'learning_rate': 0.005, 'hidden_units': 128, 'num_blocks': 3, 'dropout_rate': 0.2,
     'num_heads': 1, 'l2_emb': 0.0, 'maxlen': 200, 'batch_quota': None, 'seed': 0, 'sampler_seed': 789, 'device': None,
     'max_epochs': 8}