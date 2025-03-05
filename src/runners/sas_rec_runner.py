import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any, Union, Tuple

# import mlflow
import numpy as np
import torch
from torch import optim

from src.dataset.movielens.loader import Loader
from src.models.evaluation import Evaluator
from src.models.sas.dataset import Dataset
from src.models.sas.sasrec import SASRec
from src.models.sas.trainer import Trainer
from src.models.sas.utils.utils import get_torch_device, get_output_name
from src.models.sasrec.data.processor import prepare_data, get_sequence_length, DataSet
from src.models.sasrec.learning import train_data_format, test_data_format
from src.models.sasrec.m_model import train_validate
from src.models.sasrec.utils import save_config, dump_trial_results
from src.runners.abstract_runnner import AbstractRunner
from src.utils.enums import MovieLensDataset, TargetMetric
from src.utils.logger import Logger
logger = logging.getLogger()

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

    def run_experiment(self, datapack, config):
        """
        Executa o treinamento e avaliação do modelo utilizando os dados e a configuração fornecidos.

        Args:
            datapack: Dados de entrada para o experimento.
            run_args: Parâmetros de execução (argumentos da linha de comando).
            config: Dicionário com a configuração (hiperparâmetros e demais parâmetros).

        Returns:
            score: Valor do score obtido na métrica alvo.
            results: Resultados completos da avaliação.
            evaluator: Instância do avaliador (Evaluator) após a execução.
        """
        # Cria a instância do dataset e inicializa os formatos necessários
        dataset = DataSet(
            datapack,
            name=self.dataset,
            train_format=train_data_format(self.model),
            test_format=test_data_format(self.next_item_only)
        )
        dataset.initialize_formats({'train': 'sequential'})
        dataset.info()

        # Prepara a rotina de treinamento do modelo
        evaluator = Evaluator(dataset, self.topn)

        logger.info(f'Starting training with configuration: {config}')
        logger.info(f'Target metric: {self.target_metric.value.upper()}@{self.topn}')

        # Executa o treinamento e avaliação
        train_validate(config, evaluator)
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
        tune_datapack, test_datapack = prepare_data(self.dataset, time_offset_q=[self.time_offset] * 2)
        config = return_config()
        if config.get('maxlen') is None:
            config['maxlen'] = get_sequence_length()

        if self.save_config:
            save_config(config=config, experiment_name=study_name)

        score, results, evaluator = self.run_experiment(tune_datapack,  config)
        logger.info(f"Training score: {score}")

        # Caso a flag de teste esteja ativa, executa a avaliação no conjunto de teste
        if self.check_best:
            logger.info("Running test evaluation on best configuration...")
            test_score, test_results, test_evaluator = self.run_experiment(test_datapack, config)
            logger.info(f"Test results for the provided parameters:\n{test_results}")
            logger.info(test_evaluator.results)
            if self.dump_results:
                dump_trial_results(test_evaluator.results, config, f'{study_name}_TEST')

    pass


def return_config() -> dict:
    return {'batch_size': 256, 'learning_rate': 0.005, 'hidden_units': 128, 'num_blocks': 3, 'dropout_rate': 0.2,
     'num_heads': 1, 'l2_emb': 0.0, 'maxlen': 200, 'batch_quota': None, 'seed': 0, 'sampler_seed': 789, 'device': None,
     'max_epochs': 4}