from six.moves import range
from six.moves import zip

import random

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
from ..typedef import *

class Iterator(object):
    """Base Iterator class."""
    __metaclass__ = ABCMeta

    @staticmethod
    def mask_data(seqs):
        """Pads sequences with EOS (0) for minibatch processing."""
        lengths = [len(s) for s in seqs]
        n_samples = len(seqs)

        maxlen = np.max(lengths) + 1

        # Shape is (t_steps, samples)
        x = np.zeros((maxlen, n_samples)).astype(INT)
        x_mask = np.zeros_like(x).astype(FLOAT)

        for idx, s_x in enumerate(seqs):
            x[:lengths[idx], idx] = s_x
            x_mask[:lengths[idx] + 1, idx] = 1.

        return x, x_mask

    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None):
        self.batch_size = batch_size
        self.n_samples  = 0
        self.seed       = seed
        self.mask       = mask
        self._keys     = []
        self._idxs     = []
        self._seqs     = []
        self._iter     = None
        self._minibatches = []

        self.shuffle_mode = shuffle_mode
        if self.shuffle_mode:
            # Set random seed
            random.seed(self.seed)

    def __len__(self):
        """Returns number of samples."""
        return self.n_samples

    def __iter__(self):
        return self

    def next(self):
        """Returns the next set of data from the iterator."""
        try:
            data = self.get_batch_list()
        except StopIteration as si:
            self.rewind()
            raise
        else:
            # Lookup the keys and return an ordered dict of the current minibatch
            return OrderedDict([(k, data[i]) for i,k in enumerate(self._keys)])

    def get_batch_list(self):
        return next(self._iter)

    @abstractmethod
    def read(self):
        """Read the data and put in into self.__seqs."""
        pass

    @abstractmethod
    def prepare_batches(self):
        """Prepare self.__iter."""
        pass

    @abstractmethod
    def rewind(self):
        pass
