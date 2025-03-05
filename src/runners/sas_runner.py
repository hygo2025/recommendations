import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any, Union, Tuple

import mlflow
import numpy as np
import torch
from torch import optim

from src.dataset.movielens.loader import Loader
from src.models.sas.dataset import Dataset
from src.models.sas.sasrec import SASRec
from src.models.sas.trainer import Trainer
from src.models.sas.utils.utils import get_torch_device, get_output_name
from src.runners.abstract_runnner import AbstractRunner
from src.utils.enums import MovieLensDataset
from src.utils.logger import Logger


class SasRunner(AbstractRunner):
    def __init__(self,
        debug: bool = False,
        save: bool = False,
        log_dir: str = "../logs",
        random_seed: int = 42,
        resume_dir: str = "",
        max_seq_len: int = 50,
        batch_size: int = 128,
        data_root: str = "../data",
        hidden_dim: int = 50,
        num_blocks: int = 2,
        dropout_p: float = 0.5,
        share_item_emb: bool = False,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        evaluate_k: int = 10,
        num_epochs: int = 2000,
        early_stop_epoch: int = 20,
        use_scheduler: bool = False,
        warmup_ratio: float = 0.05,
        scheduler_type: str = "onecycle",
        resume_training: bool = False,
        output_dir: str = "../outputs",
        mlflow_experiment: str = "sasrec-pytorch-experiments",
        mlflow_run_name: str = "",
    ):
        self.logger = Logger.get_logger(name="SarRunner")
        self.loader = Loader()
        self.debug = debug
        self.save = save
        self.log_dir = log_dir
        self.random_seed = random_seed
        self.resume_dir = resume_dir
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.data_root = data_root
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout_p = dropout_p
        self.share_item_emb = share_item_emb
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.evaluate_k = evaluate_k
        self.num_epochs = num_epochs
        self.early_stop_epoch = early_stop_epoch
        self.use_scheduler = use_scheduler
        self.warmup_ratio = warmup_ratio
        self.scheduler_type = scheduler_type
        self.resume_training = resume_training
        self.output_dir = output_dir
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_run_name = mlflow_run_name
        self.device = get_torch_device()
        self.dataset_type: MovieLensDataset = MovieLensDataset.ML_100K


    def run(self) -> None:
        torch.manual_seed(self.random_seed)
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

        mlflow.set_experiment(experiment_name=self.mlflow_experiment)

        args = defaultdict(list)

        # Get timestamp.
        time_right_now = time.time()
        timestamp = datetime.fromtimestamp(time_right_now)
        timestamp = timestamp.strftime("%m-%d-%Y-%H%M")
        args.timestamp = timestamp

        save_dir, log_filepath = self.setup_training_parameters(
            resume_training=False,
            output_dir=self.output_dir,
            log_dir=self.log_dir,
            timestamp=timestamp,
            data_name=self.dataset_type.value,
        )

        # ----------------- #

        dataset = Dataset(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            dataset_type=MovieLensDataset.ML_100K,
        )

        args.num_items = dataset.num_items

        model = SASRec(
            num_items=dataset.num_items,
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.max_seq_len,
            dropout_p=self.dropout_p,
            share_item_emb=self.share_item_emb,
            device=self.device,
        )

        model = model.to(self.device)


        optimizer = optim.Adam(
            params=model.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        trainer = Trainer(
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            device=self.device,
            evaluate_k=self.evaluate_k,
            max_lr=self.lr,
            num_epochs=self.num_epochs,
            early_stop_epoch=self.early_stop_epoch,
            use_scheduler=self.use_scheduler,
            warmup_ratio=self.warmup_ratio,
            scheduler_type=self.scheduler_type,
            resume_training=self.resume_training,
            save_dir=save_dir,
        )

        mlflow.end_run()

        with mlflow.start_run(run_name=self.mlflow_run_name):
            mlflow.log_artifact(local_path=log_filepath, artifact_path="logs")
            best_results = trainer.train()
            best_ndcg_epoch, best_model_state_dict, _ = best_results

            # Perform test.
            model.load_state_dict(best_model_state_dict)
            self.logger.info(f"Testing with model checkpoint from epoch {best_ndcg_epoch}...")
            test_ndcg, test_hit_rate = trainer.evaluate(mode="test", model=model)

            test_ndcg_msg = f"Test nDCG@{self.evaluate_k} is {test_ndcg: 0.6f}."
            test_hit_msg = f"Test Hit@{self.evaluate_k} is {test_hit_rate: 0.6f}."
            test_result_msg = "\n".join([test_ndcg_msg, test_hit_msg])
            self.logger.info(f"\n{test_result_msg}")


    def setup_training_parameters(
            self,
            resume_training: bool,
            output_dir: str,
            log_dir: str,
            timestamp: str,
            data_name: str,
            resume_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Configura os diretórios de salvamento e log para o treinamento.

        Se o treinamento está sendo retomado, utiliza o diretório informado (ou o mais recente encontrado)
        para definir o caminho de salvamento e log. Caso contrário, gera um novo nome de saída, cria os diretórios
        necessários e salva os parâmetros em um arquivo 'args.json'.

        Retorna:
            Tuple[str, str]: Caminho de salvamento e caminho do log.
        """
        if resume_training:
            if resume_dir:
                log_filepath = f"{resume_dir.replace(output_dir, log_dir)}.log"
                return resume_dir, log_filepath
            else:
                # Procura pelo diretório mais recente em output_dir que contenha data_name
                relevant_files = [f for f in os.listdir(output_dir) if data_name in f]
                if not relevant_files:
                    raise ValueError("Nenhum diretório relevante encontrado em output_dir para o data_name informado.")
                timestamps = [f.split("_")[-1] for f in relevant_files]
                timestamp_objs = [
                    datetime.strptime(ts, "%m-%d-%Y-%H%M").timestamp() for ts in timestamps
                ]
                most_recent_ts_idx = int(np.argmax(timestamp_objs))
                resume_dir_found = relevant_files[most_recent_ts_idx]
                full_resume_dir = os.path.join(output_dir, resume_dir_found)
                log_filepath = os.path.join(log_dir, f"{resume_dir_found}.log")
                return full_resume_dir, log_filepath

        else:
            os.makedirs(log_dir, exist_ok=True)
            # Gera o nome de saída usando a função get_output_name (que depende dos parâmetros do objeto)
            output_name = get_output_name(
                data_filename=data_name,
                lr=self.lr,
                batch_size=self.batch_size,
                early_stop_epoch=self.early_stop_epoch,
                num_epochs=self.num_epochs,
                random_seed=self.random_seed,
                timestamp=timestamp
            )
            log_filename = f"{output_name}.log"
            log_filepath = os.path.join(log_dir, log_filename)
            save_dir = os.path.join(output_dir, output_name)
            os.makedirs(save_dir, exist_ok=True)

            # Salva os parâmetros em um arquivo JSON
            args_save_filename = os.path.join(save_dir, "args.json")
            params = {
                "output_dir": output_dir,
                "log_dir": log_dir,
                "mlflow_run_name": output_name,
                "log_filepath": log_filepath,
                "save_dir": save_dir,
                "data_name": data_name
            }
            with open(args_save_filename, "w") as f:
                json.dump(params, f, indent=2)

            return save_dir, log_filepath


