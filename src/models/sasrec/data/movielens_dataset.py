from contextlib import contextmanager
import numpy as np
import pandas as pd

from src.models.sasrec import defaults
from src.utils.enums import MovieLensDataset
from src.utils.logger import Logger


class MovielensDataSet:
    """
    Classe responsável por gerenciar os dados e suas transformações em diferentes formatos para treinamento e teste.

    Atributos:
        _data_container: dicionário que armazena os dados originais ('default') e os dados transformados.
        default_formats: dicionário com os formatos padrão para 'train' e 'test'.
        is_persistent: flag que indica se os dados transformados devem ser cacheados.
        name: nome do dataset.
    """

    def __init__(self, datapack, train_format: str = 'sequential_packed', test_format: str = 'sequential',
                 is_persistent: bool = True, dataset_name: MovieLensDataset = None):
        """
        Inicializa o DataSet com os dados e configura os formatos padrão.

        Parâmetros:
            datapack: tupla contendo (train, test, index).
            train_format: formato padrão para os dados de treinamento.
            test_format: formato padrão para os dados de teste.
            is_persistent: se True, os dados transformados serão cacheados.
            name: nome opcional do dataset.
        """
        train, test, index = datapack
        self._index = index
        self._data_container = {
            'default': {
                'train': train,
                'test': test,
            }
        }
        self.default_formats = {'train': train_format, 'test': test_format}
        self.is_persistent = is_persistent
        self.logger = Logger.get_logger(name="DataSet")
        self.initialize_formats(self.default_formats)
        self.name = dataset_name if dataset_name is not None else self.__class__.__name__

    def initialize_formats(self, formats: dict):
        """
        Garante que os dados nos formatos especificados estão disponíveis.

        Parâmetros:
            formats: dicionário com os modos ('train' e 'test') e seus respectivos formatos.
        """
        for mode, fmt in formats.items():
            formatted = self.get_formatted_data(mode, fmt)
            if formatted is None:
                raise ValueError(f"Falha ao inicializar o formato '{fmt}' para o modo '{mode}'.")

    @property
    def user_index(self):
        """Retorna o índice de usuários."""
        return self._index['users']

    @property
    def item_index(self):
        """Retorna o índice de itens."""
        return self._index['items']

    @property
    def train(self):
        """Retorna os dados de treinamento no formato padrão."""
        return self.get_formatted_data('train', self.default_formats['train'])

    @property
    def test(self):
        """Retorna os dados de teste no formato padrão."""
        return self.get_formatted_data('test', self.default_formats['test'])

    def _dataframe_to_sequences(self, data: pd.DataFrame, userid: str, itemid: str, timeid: str):
        """
        Converte um DataFrame de interações em uma Série, onde cada usuário possui uma lista de itens ordenados pelo tempo.
        """
        self.logger.debug("Convertendo dataframe para formato 'sequential'.")
        sorted_data = data.sort_values(timeid)
        sequences = sorted_data.groupby(userid, sort=False)[itemid].apply(list)
        return sequences

    def _dataframe_to_packed_sequences(self, data: pd.DataFrame, userid: str, itemid: str, timeid: str):
        """
        Converte um DataFrame de interações em uma tupla (indices, sizes) para uso em modelos sequenciais.

        Retorna:
            indices: array com todos os itens concatenados.
            sizes: array com os tamanhos cumulativos de cada sequência de usuário.

        Levanta NotImplementedError se os índices de usuários não seguirem uma sequência contínua de 0 a N-1.
        """
        self.logger.debug("Convertendo dataframe para formato 'sequential_packed'.")
        sorted_data = data.sort_values(timeid)
        sequences = sorted_data.groupby(userid, sort=True)[itemid].apply(list)
        num_users = sequences.index.max() + 1

        if len(sequences) != num_users:
            msg = "Apenas índices contínuos `0,1,...,N-1` são suportados."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        sizes = np.zeros(num_users + 1, dtype=np.intp)
        sizes[1:] = sequences.apply(len).cumsum().values
        indices = np.concatenate(sequences.to_list(), dtype=np.int32)
        return indices, sizes

    def _get_transformation_function(self, fmt: str):
        """
        Retorna a função de transformação correspondente ao formato desejado.

        Parâmetros:
            fmt: formato (por exemplo, 'sequential' ou 'sequential_packed').

        Retorna:
            Função de transformação.
        """
        transformation_mapping = {
            'sequential': self._dataframe_to_sequences,
            'sequential_packed': self._dataframe_to_packed_sequences,
        }
        if fmt not in transformation_mapping:
            msg = f"Formato não reconhecido: {fmt}"
            self.logger.error(msg)
            raise NotImplementedError(msg)
        return transformation_mapping[fmt]

    def get_formatted_data(self, mode: str, fmt: str = None):
        """
        Recupera os dados formatados para o modo especificado (train ou test).

        Se os dados já estiverem cacheados, retorna-os; caso contrário, realiza a transformação.

        Parâmetros:
            mode: 'train' ou 'test'.
            fmt: formato desejado. Se None, usa o formato padrão para o modo.

        Retorna:
            Dados formatados.
        """
        if fmt is None:
            fmt = self.default_formats.get(mode)
        data_container = self._data_container.setdefault(fmt, {})

        if mode in data_container:
            self.logger.debug(f"Retornando dados cacheados para {mode} no formato {fmt}.")
            return data_container[mode]

        self.logger.debug(f"Transformando dados {mode} para o formato {fmt}.")
        raw_data = self._data_container['default'][mode]
        userid = self.user_index.name
        itemid = self.item_index.name
        timeid = defaults.timeid

        transform_func = self._get_transformation_function(fmt)
        formatted_data = transform_func(raw_data, userid, itemid, timeid)

        if self.is_persistent:
            data_container[mode] = formatted_data

        return formatted_data

    @contextmanager
    def formats(self, train: str = None, test: str = None):
        """
        Context manager para alterar temporariamente os formatos dos dados.

        Parâmetros:
            train: formato temporário para os dados de treinamento.
            test: formato temporário para os dados de teste.

        Exemplo:
            with dataset.formats(train='sequential', test='sequential_packed'):
                # operações com os novos formatos
        """
        current_train = self.default_formats['train']
        current_test = self.default_formats['test']
        self.default_formats = {
            'train': train or current_train,
            'test': test or current_test
        }
        try:
            yield self
        finally:
            self.default_formats = {'train': current_train, 'test': current_test}

    def format_exists(self, mode: str, fmt: str) -> bool:
        """
        Verifica se o formato especificado já foi computado para o modo dado.

        Parâmetros:
            mode: 'train' ou 'test'.
            fmt: formato a ser verificado.

        Retorna:
            True se existir, False caso contrário.
        """
        return fmt in self._data_container and mode in self._data_container[fmt]

    def info(self):
        """
        Exibe informações e estatísticas principais do dataset.
        """
        self.logger.info(f'Formatos do dataset: {self.default_formats}.')
        test_data = self._data_container['default']['test']
        userid = self.user_index.name
        itemid = self.item_index.name
        test_stats = test_data[[userid, itemid]].nunique()
        self.logger.info(
            f"O dataset {self.name} possui {test_data.shape[0]} interações de teste entre "
            f"{test_stats[userid]} usuários e {test_stats[itemid]} itens."
        )
