from contextlib import contextmanager

import numpy as np

from src.models.sasrec import defaults
from src.utils.logger import Logger


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

