import torch
from typing import Tuple, List, Dict

from tqdm import tqdm

from src.dataset.movielens.loader import Loader
from src.models.sas.utils.utils import get_positive2negatives
from src.utils.enums import MovieLensDataset, MovieLensType
from collections import defaultdict

User = str
Item = str
InputSequences = torch.Tensor
PositiveSamples = torch.Tensor
NegativeSamples = torch.Tensor
ItemIdxs = torch.Tensor


class Dataset:
    def __init__(
            self,
            batch_size: int,
            max_seq_len: int,
            dataset_type: MovieLensDataset
    ):
        self.loader = Loader()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.data = self.loader.load_pandas(dataset=dataset_type, ml_type=MovieLensType.RATINGS)[['userId', 'movieId']].values.tolist()
        self.user2items, self.item2users = self.create_mappings(data=self.data)
        self.num_users = len(self.user2items)
        self.num_items = len(self.item2users)
        self.positive2negatives = get_positive2negatives(self.num_items)
        self.user2items_train, self.user2items_valid, self.user2items_test = self.create_train_valid_test(user2items=self.user2items)

    def create_mappings(
            self, data: List[Tuple[User, Item]]
    ) -> Tuple[Dict[User, List[Item]], Dict[Item, List[User]]]:
        """
        Converte uma lista de pares (user, item) em dois dicionários:
          - Um mapeando cada usuário para a lista de items associados;
          - Outro mapeando cada item para a lista de usuários associados.
        """
        user2items = defaultdict(list)
        item2users = defaultdict(list)

        for user, item in tqdm(data, desc="Creating user2items", total=len(data)):
            user2items[user].append(item)
            item2users[item].append(user)

        return dict(user2items), dict(item2users)

    def create_train_valid_test(
            self, user2items: Dict[User, List[Item]]
    ) -> Tuple[
        Dict[User, List[Item]],
        Dict[User, Tuple[List[Item], Item]],
        Dict[User, Tuple[List[Item], Item]]
    ]:
        """
        Cria os conjuntos de treino, validação e teste para cada usuário.

        Para usuários com menos de três itens, todas as interações são usadas no treino,
        enquanto os conjuntos de validação e teste permanecem vazios.
        Para usuários com três ou mais interações:
          - O conjunto de treino é composto por todos os itens, exceto os dois últimos.
          - O conjunto de validação é uma tupla, onde o primeiro elemento é a sequência de itens
            de treino (itens[:-2]) e o segundo elemento é o penúltimo item (itens[-2]).
          - O conjunto de teste é uma tupla, onde o primeiro elemento é a sequência formada pela
            concatenação da sequência de treino com o penúltimo item, e o segundo elemento é o
            último item (itens[-1]).
        """
        user2items_train = {}
        user2items_valid = {}
        user2items_test = {}

        for user, items in tqdm(user2items.items(), desc="Getting train/valid/test splits", total=len(user2items)):
            if len(items) < 3:
                user2items_train[user] = items
                user2items_valid[user] = []
                user2items_test[user] = []
            else:
                train_seq = items[:-2]
                user2items_train[user] = train_seq
                user2items_valid[user] = (train_seq, items[-2])
                user2items_test[user] = (train_seq + [items[-2]], items[-1])

        return user2items_train, user2items_valid, user2items_test

