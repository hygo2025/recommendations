import os
import re
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.models.sasrec import defaults
from src.models.sasrec.utils import reindex
from src.utils.logger import Logger
from . import movielens as ml

class DataSet:
    def __init__(self, datapack, train_format='sequential_packed', test_format='sequential', is_persistent=True, name=None):
        train, test, index = datapack
        self._index = index
        self._data_container = {
            'default': {
                'train': train, # model specific
                'test': test,   # experiment specific
            }
        }
        self.default_formats = {'train': train_format, 'test': test_format}
        self._format_kwargs = {}
        self.is_persistent = is_persistent # cache calculated data formats
        self.initialize_formats(self.default_formats)
        self.name = name if name is not None else self.__class__.__name__
        self.logger = Logger.get_logger(name="DataSet")


    def initialize_formats(self, formats):
        for mode, format in formats.items():
            assert self.get_formatted_data(mode, format) is not None

    @property
    def user_index(self):
        return self._index['users']

    @property
    def item_index(self):
        return self._index['items']

    @property
    def train(self):
        return self.get_formatted_data('train', self.default_formats['train'])

    @property
    def test(self):
        return self.get_formatted_data('test', self.default_formats['test'])

    def get_formatted_data(self, mode, format=None):
        data_container = self._data_container.setdefault(format, {})
        if mode in data_container:
            formatted_data = data_container[mode]
        else:
            data = self._data_container['default'][mode]
            userid = self.user_index.name
            itemid = self.item_index.name
            timeid = defaults.timeid

            if format == 'sequential':
                formatted_data = dataframe_to_sequences(data, userid, itemid, timeid)
            elif format == 'sequential_packed':
                formatted_data = dataframe_to_packed_sequences(data, userid, itemid, timeid)
            else:
                raise NotImplementedError(f'Unrecognized format: {format}')

            if self.is_persistent:
                data_container[mode] = formatted_data

        return formatted_data

    @contextmanager
    def formats(self, train=None, test=None):
        # store current values of data formats
        train_format = self.default_formats['train']
        test_format = self.default_formats['test']
        self.default_formats = { # temporarily set new data formats
            'train': train or train_format, # if not set - use current format
            'test': test or test_format # if not set - use current format
        }
        try:
            yield self
        finally: # restore initial values
            self.default_formats = {'train': train_format, 'test': test_format}

    def format_exists(self, mode, format):
        try:
            data = self._data_container[format]
        except KeyError:
            return False
        return mode in data

    def info(self):
        '''Display main dataset info and statistics'''
        self.logger.info(f'Dataset formats: {self.default_formats}.')
        test_data = self._data_container['default']['test']
        # TODO stepwise interactions may lack some data from default format
        userid = self.user_index.name
        itemid = self.item_index.name
        test_stats = test_data[[userid, itemid]].nunique()
        self.logger.info(
            f'Test data from {self.name} contains {test_data.shape[0]} interactions '
            f'between {test_stats[userid]} users and {test_stats[itemid]} items.'
        )

def dataframe_to_sequences(data, userid, itemid, timeid):
    '''Convert observations dataframe into a user-keyed series of lists of item sequences.'''
    return data.sort_values(timeid).groupby(userid, sort=False)[itemid].apply(list)

def dataframe_to_packed_sequences(data, userid, itemid, timeid):
    sequences = data.sort_values(timeid).groupby(userid, sort=True)[itemid].apply(list)
    num_users = sequences.index.max() + 1
    if len(sequences) != num_users:
        raise NotImplementedError('Only continuous `0,1,...,N-1` index of N users is supported')
    sizes = np.zeros(num_users + 1, dtype=np.intp)
    sizes[1:] = sequences.apply(len).cumsum().values
    indices = np.concatenate(sequences.to_list(), dtype=np.int32)
    return indices, sizes


def prepare_data(dataset_name, time_offset_q=None):
    time_offset_valid, time_offset_test = read_time_offsets(time_offset_q)
    userid, itemid, timeid = entity_names()
    data = read_raw_data(dataset_name)
    
    if time_offset_valid is None:
        train_data_, test_data_ = split_data_by_time(
            data, time_offset_test, timeid, max_samples=defaults.max_test_interactions
        )
        test_datapack = reindex_data(train_data_.copy(), test_data_, userid, itemid)
        return (test_datapack,)

    eval_offset = time_offset_test + time_offset_valid - 1
    valid_ratio = (1 - time_offset_valid) / (1 - eval_offset)
    max_valid_ratio = defaults.max_test_interactions / (len(data) * (1 - eval_offset))
    if valid_ratio > max_valid_ratio: # extend the offset to preserve valid/test data size ratio
        eval_offset = 1 - defaults.max_test_interactions / (len(data) * valid_ratio)

    train_data_valid_, rest_data_ = split_data_by_time(data, eval_offset, timeid)
    valid_data_, test_data_ = split_data_by_time(rest_data_, valid_ratio, timeid)

    tune_datapack = reindex_data(
        train_data_valid_.copy(), valid_data_, userid, itemid
    )
    train_data_ = pd.concat([train_data_valid_, valid_data_], axis=0)
    test_datapack = reindex_data(
        train_data_.copy(), test_data_, userid, itemid
    )
    return tune_datapack, test_datapack

def read_time_offsets(time_offset_q):
    '''
    Always returns (validation, test) offsets tuple.
    '''
    if isinstance(time_offset_q, (list, tuple)):
        time_offset_valid, time_offset_test = time_offset_q
        return time_offset_valid, time_offset_test
    return None, time_offset_q or defaults.time_offset_q


def read_raw_data(dataset_name):
    # data_path = os.path.join(defaults.data_dir, f'/raw/{dataset_name.name.lower()}.gz')
    data_path = f'{defaults.data_dir}/raw/{dataset_name.name.lower()}.gz'
    return pd.read_csv(data_path, na_filter=False)  # assume data is clean (mostly for steam data)


def entity_names():
    userid = defaults.userid
    timeid = defaults.timeid
    itemid = 'movieid'

    return userid, itemid, timeid


def split_data_by_time(data, time_q, timeid, max_samples=None):
    test_timepoint = data[timeid].quantile(q=time_q, interpolation='nearest')
    test_time = data[timeid] >= test_timepoint
    test_data = data.loc[test_time, :]
    if max_samples is not None and len(test_data) > max_samples:
        # If the number of rows in test_data exceeds max_samples,
        # take only the latest rows based on timeid
        test_data = test_data.sort_values(by=timeid, ascending=True).tail(max_samples)
        train_data = data.drop(test_data.index)
    else:
        train_data = data.loc[~test_time, :]
    return train_data, test_data


def reindex_data(train, test, userid, itemid, verbose=False):
    train_data, data_index = transform_indices(train, userid, itemid)
    # reindex items (and exclude cold-start items)
    test_data = reindex(test, data_index['items'], filter_invalid=True)
    # reindex users
    test_user_idx = data_index['users'].get_indexer(test_data[userid])
    is_new_user = test_user_idx == -1
    if is_new_user.any(): # track unseen users - to be used in warm-start regime
        new_user_idx, data_index['new_users'] = pd.factorize(test_data.loc[is_new_user, userid])
        # ensure no intersection with train users index
        test_user_idx[is_new_user] = new_user_idx + len(data_index['users'])
    # assign new user index
    test_data.loc[:, userid] = test_user_idx
    return train_data, test_data, data_index


def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        codes, index = to_numeric_id(data, field)
        data_index[entity] = index
        data.loc[:, field] = codes
    return data, data_index


def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    codes = idx_data.cat.codes
    index = idx_data.cat.categories.rename(field)
    return codes, index