import copy
import logging
import torch
from datetime import datetime
from typing import Optional

from tqdm import tqdm
import numpy as np

from src.models.evaluation import Evaluator
from src.models.sasrec import defaults
from src.models.sasrec.data.movielens_data_setup import MovieLensDataSetup
from src.models.sasrec.data.movielens_dataset import MovielensDataSet
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


class EarlyStopping(TrialStopException):
    pass


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

    def _load_adjusted_state_dict(self, model, state_dict):
        """
        Ajusta o state_dict para que o tamanho do embedding de itens seja compatível com o modelo atual.
        Caso haja size mismatch na chave 'item_emb.weight', os pesos do checkpoint serão copiados para os índices correspondentes
        e os itens extras manterão os pesos iniciais.
        """
        key = "item_emb.weight"
        if key in state_dict:
            ckpt_weight = state_dict[key]
            current_weight = model.item_emb.weight
            if ckpt_weight.shape != current_weight.shape:
                self.logger.info(
                    f"Ajustando tamanho do '{key}': checkpoint {ckpt_weight.shape} -> atual {current_weight.shape}"
                )
                # Cria uma nova matriz com os pesos atuais
                new_weight = current_weight.data.clone()
                n_copy = min(ckpt_weight.shape[0], current_weight.shape[0])
                new_weight[:n_copy] = ckpt_weight[:n_copy]
                state_dict[key] = new_weight
        # Carrega o state_dict ajustado de forma estrita (agora devem bater as dimensões)
        model.load_state_dict(state_dict, strict=True)

    def _train_model(self, datapack, config):
        """
        Treina o modelo no conjunto de dados de treino/validação e retorna:
         - best_model_state: pesos do melhor modelo (usando deep copy)
         - score: score obtido com o target metric no melhor ponto de validação
         - results: resultados completos da última avaliação
         - evaluator: instância do avaliador utilizada
         - n_items: número de itens (para criação do modelo)
         - dataset: instância do MovielensDataSet utilizada
        """
        # Instancia o dataset e inicializa os formatos
        dataset = MovielensDataSet(
            datapack,
            dataset_name=self.dataset,
            train_format='sequential_packed',
            test_format='sequential'
        )
        dataset.initialize_formats({'train': 'sequential'})
        dataset.info()

        evaluator = Evaluator(dataset, self.topn)
        self.logger.info(f"Starting training with configuration: {config}")
        self.logger.info(f"Target metric: {self.target_metric.value.upper()}@{self.topn}")

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
        if validation_interval > max_epochs:
            raise ValueError("Número de épocas é muito pequeno para realizar validação.")

        best_score = -np.inf
        best_epoch = None
        best_model_state = None

        for epoch in range(max_epochs):
            self.logger.info(f"Processando época {epoch + 1}/{max_epochs}")
            loss = model.train_epoch(sampler, n_batches)
            if (epoch + 1) % validation_interval == 0:
                try:
                    evaluator.submit(model, step=epoch, args=(loss,))
                    # Recupera os resultados da avaliação recente
                    curr_results = evaluator.most_recent_results
                    curr_score = curr_results.loc[f'{self.target_metric.value.upper()}@{self.topn}', 'score']
                    self.logger.info(f"Época {epoch + 1}: Loss = {loss:.4f}, Validação = {curr_score:.4f}")
                    if curr_score > best_score:
                        best_score = curr_score
                        best_epoch = epoch
                        best_model_state = copy.deepcopy(model.model.state_dict())
                        self.logger.info(f"Melhor modelo atualizado na época {epoch + 1} com score {curr_score:.4f}")
                except EarlyStopping:
                    self.logger.info("Early stopping acionado.")
                    break

        if best_model_state is None:
            self.logger.warning("Nenhum modelo foi salvo durante o treinamento; usando o estado final do modelo.")
            best_model_state = model.model.state_dict()

        # Tenta recuperar os resultados associados ao melhor modelo
        try:
            best_results = evaluator.results['best']
            results = evaluator.results[best_results['step']]
        except KeyError:
            results = evaluator.most_recent_results

        score = results.loc[f'{self.target_metric.value.upper()}@{self.topn}', 'score']
        return best_model_state, score, results, evaluator, n_items, dataset

    def _evaluate_model(self, datapack, config, model_state):
        """
        Avalia o modelo no conjunto de teste.
        Retorna:
         - score: score obtido no conjunto de teste
         - results: resultados completos da avaliação
         - evaluator: instância do avaliador utilizada
         - dataset: instância do MovielensDataSet utilizada
         - model: o modelo carregado com os melhores pesos
        """
        dataset = MovielensDataSet(
            datapack,
            dataset_name=self.dataset,
            train_format='sequential_packed',
            test_format='sequential'
        )
        dataset.info()
        evaluator = Evaluator(dataset, self.topn)
        n_items = len(dataset.item_index)
        model = SASRecModel(config, n_items)    # Ajusta e carrega o state_dict
        self._load_adjusted_state_dict(model.model, model_state)
        self.logger.info("Iniciando avaliação no conjunto de teste...")
        evaluator.submit(model, step=0, args=(None,))
        results = evaluator.most_recent_results
        score = results.loc[f'{self.target_metric.value.upper()}@{self.topn}', 'score']
        return score, results, evaluator, dataset, model

    def run(self) -> None:
        ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        study_name = f'{self.model}_{self.dataset}_{self.target_metric}_{ts}'
        # Prepara os conjuntos de dados (treino/validação e teste)
        tune_datapack, test_datapack = MovieLensDataSetup(
            self.dataset, time_offset_q=[self.time_offset] * 2
        ).prepare_data()

        config = return_config()
        if config.get('maxlen') is None:
            config['maxlen'] = defaults.sequence_length_movies

        if self.save_config:
            save_config(config=config, experiment_name=study_name)

        # Fase de treinamento
        best_model_state, train_score, train_results, train_evaluator, n_items, train_dataset = self._train_model(tune_datapack, config)
        self.logger.info(f"Score de treinamento: {train_score:.4f}")

        if best_model_state is not None:
            best_model_path = f"/home/hygo/Development/recommendations/data/results/{study_name}_best_model.pth"
            torch.save(best_model_state, best_model_path)
            self.logger.info(f"Pesos do melhor modelo salvos em {best_model_path}")
        else:
            self.logger.warning("Nenhum estado de melhor modelo foi capturado.")

        # Fase de avaliação no conjunto de teste
        if self.check_best:
            test_score, test_results, test_evaluator, test_dataset, best_model = self._evaluate_model(test_datapack, config, best_model_state)
            self.logger.info(f"Score no conjunto de teste: {test_score:.4f}")
            self.logger.info(f"Resultados no conjunto de teste:\n{test_results}")
            if self.dump_results:
                dump_trial_results(test_evaluator.results, config, f'{study_name}_TEST')

            # Exemplo de recomendação para um usuário (usando o primeiro usuário do dataset de teste)
            try:
                sample_user = test_dataset.user_index[0]
                recommendations = best_model.recommend(sample_user, self.topn) #TODO Aqui ta dando erro
                self.logger.info(f"Recomendações para o usuário {sample_user}: {recommendations}")
            except Exception as e:
                self.logger.error(f"Erro ao gerar recomendação de exemplo: {e}")


def test_data_format(next_item_only):
    if next_item_only:  # e.g., lorentzfm não é adequado para predição sequencial
        return ('interactions', dict(stepwise=True, max_steps=1))
    return 'sequential'


def train_data_format(model_name):
    sparse = ['svd', 'random', 'mostpop']
    sequential_packed = ['sasrec', 'sasrecb', 'hypsasrec', 'hypsasrecb']
    sequential = []
    sequential_typed = []
    if model_name in sparse:
        return 'sparse'
    if model_name in sequential:
        return 'sequential'  # pandas Series
    if model_name in sequential_packed:
        return 'sequential_packed'  # csr-like format
    if model_name in sequential_typed:
        return 'sequential_typed'  # numba dict
    return 'default'


def return_config() -> dict:
    return {
        'batch_size': 256,
        'learning_rate': 0.005,
        'hidden_units': 128,
        'num_blocks': 3,
        'dropout_rate': 0.2,
        'num_heads': 1,
        'l2_emb': 0.0,
        'maxlen': 200,
        'batch_quota': None,
        'seed': 0,
        'sampler_seed': 789,
        'device': None,
        'max_epochs': 8
    }
