import pandas as pd
import logging

from src.dataset.movielens.loader import Loader
from src.models.sasrec import defaults
from src.utils.enums import MovieLensType, MovieLensDataset


class MovieLensDataSetup:
    """
    Classe para preparar os dados do MovieLens para treinamento e avaliação.
    """

    def __init__(self, dataset: MovieLensDataset, time_offset_q=None):
        self.dataset = dataset
        self.time_offset_valid, self.time_offset_test = self._read_time_offsets(time_offset_q)
        self.idf_user = 'userid'
        self.idf_item = 'movieid'
        self.idf_time = 'timestamp'
        self.loader = Loader()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _read_time_offsets(self, time_offset_q):
        """
        Sempre retorna uma tupla (validation, test).
        """
        if isinstance(time_offset_q, (list, tuple)):
            time_offset_valid, time_offset_test = time_offset_q
            return time_offset_valid, time_offset_test
        return None, time_offset_q or defaults.time_offset_q

    def _load_data(self) -> pd.DataFrame:
        """
        Carrega os dados utilizando o Loader e renomeia as colunas relevantes.
        """
        data = self.loader.load_pandas(dataset=self.dataset, ml_type=MovieLensType.RATINGS)
        return data[['userId', 'movieId', 'timestamp']].rename(
            columns={'userId': self.idf_user, 'movieId': self.idf_item}
        )

    def _reindex(self, raw_data: pd.DataFrame, index, filter_invalid: bool = True, names=None) -> pd.DataFrame:
        """
        Realiza a fatoração dos valores das colunas com base no índice fornecido.
        Permite redefinir nomes dos índices e opcionalmente descartar linhas com entradas inválidas.
        """
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
            # -1 indica que o rótulo não está presente no índice
            cond_invalid = new_data.eval(' or '.join([f'{idx.name} == -1' for idx in index]))
            if cond_invalid.any():
                self.logger.info(f'Filtered {cond_invalid.sum()} invalid observations.')
                new_data = new_data.loc[~cond_invalid]

        return new_data

    def _split_data_by_time(self, data: pd.DataFrame, time_q: float, max_samples=None):
        """
        Divide os dados em conjuntos de treino e teste com base em um quantil temporal.
        """
        test_timepoint = data[self.idf_time].quantile(q=time_q, interpolation='nearest')
        is_test = data[self.idf_time] >= test_timepoint
        test_data = data.loc[is_test, :]

        if max_samples is not None and len(test_data) > max_samples:
            test_data = test_data.sort_values(by=self.idf_time, ascending=True).tail(max_samples)
            train_data = data.drop(test_data.index)
        else:
            train_data = data.loc[~is_test, :]

        return train_data, test_data

    def _to_numeric_id(self, data: pd.DataFrame, field: str):
        """
        Converte uma coluna para códigos numéricos utilizando categorias.
        Retorna os códigos e o índice (categorias).
        """
        cat_data = data[field].astype("category")
        codes = cat_data.cat.codes
        index = cat_data.cat.categories.rename(field)
        return codes, index

    def _transform_indices(self, data: pd.DataFrame, users: str, items: str):
        """
        Transforma os IDs de usuários e itens para índices numéricos.
        """
        data_index = {}
        for entity, field in zip(['users', 'items'], [users, items]):
            codes, index = self._to_numeric_id(data, field)
            data_index[entity] = index
            data.loc[:, field] = codes
        return data, data_index

    def _reindex_data(self, train: pd.DataFrame, test: pd.DataFrame, verbose: bool = False):
        """
        Reindexa os dados de treino e teste para garantir consistência dos índices de usuário e item.
        """
        train_data, data_index = self._transform_indices(train.copy(), self.idf_user, self.idf_item)
        test_data = self._reindex(test, data_index['items'], filter_invalid=True)
        test_user_idx = data_index['users'].get_indexer(test_data[self.idf_user])
        is_new_user = test_user_idx == -1

        if is_new_user.any():
            new_user_idx, new_users = pd.factorize(test_data.loc[is_new_user, self.idf_user])
            data_index['new_users'] = new_users
            test_user_idx[is_new_user] = new_user_idx + len(data_index['users'])

        test_data.loc[:, self.idf_user] = test_user_idx
        return train_data, test_data, data_index

    def prepare_data(self):
        """
        Prepara os dados para treinamento e avaliação.
        Caso não seja fornecido um offset de validação, retorna apenas o pacote de teste.
        Caso contrário, retorna os pacotes de afinação (tune) e teste.
        """
        data = self._load_data()

        if self.time_offset_valid is None:
            train_data, test_data = self._split_data_by_time(
                data, self.time_offset_test, max_samples=defaults.max_test_interactions
            )
            test_datapack = self._reindex_data(train_data.copy(), test_data)
            return (test_datapack,)

        eval_offset = self.time_offset_test + self.time_offset_valid - 1
        valid_ratio = (1 - self.time_offset_valid) / (1 - eval_offset)
        max_valid_ratio = defaults.max_test_interactions / (len(data) * (1 - eval_offset))
        if valid_ratio > max_valid_ratio:  # ajusta o offset para preservar a proporção entre dados válidos e de teste
            eval_offset = 1 - defaults.max_test_interactions / (len(data) * valid_ratio)

        train_data_valid, rest_data = self._split_data_by_time(data, eval_offset)
        valid_data, test_data = self._split_data_by_time(rest_data, valid_ratio)

        tune_datapack = self._reindex_data(train_data_valid.copy(), valid_data)
        train_data_combined = pd.concat([train_data_valid, valid_data], axis=0)
        test_datapack = self._reindex_data(train_data_combined.copy(), test_data)
        return tune_datapack, test_datapack
