from collections import defaultdict
from collections.abc import Iterable, Callable
from math import sqrt, log2
from typing import Optional

import numpy as np
import pandas as pd

from src.models.sasrec.data.movielens_dataset import MovielensDataSet
from src.models.sasrec.m_model.sasrec_recommender import SASRecModel


class Evaluator:
    def __init__(
            self,
            dataset: MovielensDataSet,
            topn: int,
            target_metric: Optional[str] = "NDCG",
            evaluation_callback: Optional[Callable[..., None]] = None
    ) -> None:
        """
        Parâmetros:
          - dataset: instância do MovielensDataSet
          - topn: número de recomendações
          - target_metric: métrica alvo (por exemplo, "NDCG", "HR", "MRR")
          - evaluation_callback: callback para algoritmos iterativos
        """
        self.dataset = dataset
        self.topn = topn
        self.target_metric = target_metric
        self.evaluation_callback = evaluation_callback
        self._results = {}
        self._last_used_key = None
        self.best_score = -np.inf
        self.best_step = None
        self.evaluation_time = []

    def submit(
            self,
            model: SASRecModel,
            step: Optional[int] = None,
            args: Optional[tuple] = ()
    ) -> None:
        """
        Submete o modelo para avaliação e atualiza os resultados.
        Se o score corrente for melhor do que o melhor salvo, atualiza a chave 'best'.
        """
        if self.dataset.format_exists('test', 'sequential'):
            result = evaluate_on_sequences(model, self.dataset, self.topn)
        else:
            result = evaluate(model, self.dataset, self.topn)

        self._results[step] = result
        self._last_used_key = step

        # Atualiza a melhor avaliação usando o target_metric
        if (step is not None) and (self.target_metric is not None):
            metric_key = f"{self.target_metric.upper()}@{self.topn}"
            try:
                current_score = result.loc[metric_key, 'score']
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_step = step
                    self._results['best'] = {'step': step, 'score': current_score}
            except Exception as e:
                print("Erro ao computar a melhor avaliação:", e)

        if (step is not None) and (self.evaluation_callback is not None):
            self.evaluation_callback(self._results, step, *args)

    @property
    def results(self):
        if not self._results:  # caso o evaluator ainda não tenha sido utilizado
            return None

        if self._last_used_key is None:  # para algoritmos não iterativos
            assert len(self._results) == 1, 'Algoritmos iterativos devem fornecer apenas chaves inteiras para step.'
            return self._results[None]
        return self._results  # retorna o histórico de resultados

    @property
    def most_recent_results(self):
        return self._results[self._last_used_key]


def evaluate(model: SASRecModel, dataset: MovielensDataSet, topn: int):
    with dataset.formats(train='sequential'):  # garante o formato esperado
        train_data = dataset.train
    test_data = dataset.test
    if isinstance(test_data, pd.DataFrame):  # empacota os dados num único step
        test_data = {0: test_data.itertuples(index=False, name=None)}
        # Coleta os resultados para cada step
    step_scores, step_stder2, n_unique_recs = list(zip(
        *(evaluate_step(model, train_data, test_seq, topn)
          for step, test_seq in test_data.items())
    ))
    # Calcula as médias
    average_scores = pd.DataFrame.from_records(step_scores).mean()
    average_errors = pd.DataFrame.from_records(step_stder2).mean().pow(0.5)
    average_scores.loc['COV'] = np.mean(n_unique_recs) / len(dataset.item_index)
    average_errors.loc['COV'] = sample_ci(n_unique_recs)
    averaged_results = pd.concat(
        [average_scores, average_errors],
        keys=['score', 'error'],
        axis=1,
    ).rename(index=lambda x: f'{x}@{topn}'.upper())
    return averaged_results


def evaluate_step(model: SASRecModel, train: dict, test_seq: Iterable, topn: int):
    results = []
    unique_recommendations = set()
    seen_test = defaultdict(list)
    for user, test_item in test_seq:
        seen_test_items = seen_test[user]
        seq = train.get(user, []) + seen_test_items
        if seq:
            predicted_items = model.recommend(seq, topn, user=user)
            (hit_index,) = np.where(predicted_items == test_item)
            hit_scores = compute_metrics(hit_index)
            results.append(hit_scores.values())
            unique_recommendations.update(predicted_items)
        seen_test_items.append(test_item)  # atualiza os itens vistos
    results = pd.DataFrame.from_records(results, columns=hit_scores.keys())
    step_scores = results.mean()
    step_stder2 = (results - step_scores).pow(2).mean() / (results.shape[0] - 1)
    return step_scores, step_stder2, len(unique_recommendations)


def compute_metrics(hits):
    try:
        hit_index = hits.item()
    except ValueError:  # quando não há acertos ou há mais de um
        if hits.size > 1:
            raise ValueError("Holdout deve conter um único item!")
        return {'hr': 0., 'mrr': 0., 'ndcg': 0.}
    return {'hr': 1., 'mrr': 1. / (hit_index + 1.), 'ndcg': 1. / log2(hit_index + 2.)}


def sample_ci(scores, coef=2.776):
    n = len(scores)
    if n < 2:  # não é possível estimar o intervalo de confiança
        return np.nan
    return coef * np.std(scores, ddof=1) / sqrt(n)


def evaluate_on_sequences(model, dataset, topn=10):
    with dataset.formats(train='sequential', test='sequential'):
        train_data = dataset.train
        test_data = dataset.test

    cum_hits = 0
    cum_reciprocal_ranks = 0.
    cum_discounts = 0.
    unique_recommendations = set()
    total_count = 0
    for user, test_seq in test_data.items():
        try:
            seen_seq = train_data[user]
        except KeyError:  # usuários sem histórico
            seen_seq = test_seq[:1]
            test_seq = test_seq[1:]
        num_predictions = len(test_seq)
        if not num_predictions:
            continue
        predicted_items = model.recommend_sequential(test_seq, seen_seq, topn, user=user)
        hit_steps, hit_index = np.where(predicted_items == np.atleast_2d(test_seq).T)
        unique_recommendations.update(predicted_items.ravel())

        num_hits = hit_index.size
        if num_hits:
            cum_hits += num_hits
            cum_reciprocal_ranks += np.sum(1. / (hit_index + 1))
            cum_discounts += np.sum(1. / np.log2(hit_index + 2))
        total_count += num_predictions

    hr = cum_hits / total_count
    mrr = cum_reciprocal_ranks / total_count
    dcg = cum_discounts / total_count
    cov = len(unique_recommendations) / len(dataset.item_index)
    results = pd.DataFrame(
        data={'score': [hr, mrr, dcg, cov]},
        index=[f'{metric}@{topn}' for metric in ['HR', 'MRR', 'NDCG', 'COV']]
    )
    return results
