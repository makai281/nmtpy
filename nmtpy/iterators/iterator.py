from six.moves import range
from six.moves import zip

import random

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
from ..typedef import *

class Iterator(object):
    __metaclass__ = ABCMeta

    @staticmethod
    def mask_data(seqs):
        """Pads sequences with EOS (0) for minibatch processing."""
        lengths = [len(s) for s in seqs]
        n_samples = len(seqs)

        # TODO: For ff-enc, we will need fixed tsteps in the input
        maxlen = np.max(lengths) + 1

        # Shape is (t_steps, samples)
        x = np.zeros((maxlen, n_samples)).astype(INT)
        x_mask = np.zeros_like(x).astype(FLOAT)

        for idx, s_x in enumerate(seqs):
            x[:lengths[idx], idx] = s_x
            x_mask[:lengths[idx] + 1, idx] = 1.

        return x, x_mask

    def __init__(self, batch_size, seed=1234, mask=True,
                 shuffle_mode=None):
        self.batch_size = batch_size
        self.n_samples  = 0
        self.seed       = seed
        self.mask       = mask
        self._keys     = []
        self._idxs     = []
        self._seqs     = []
        self._iter     = None

        self.shuffle_mode = shuffle_mode
        if self.shuffle_mode:
            # Set random seed
            random.seed(self.seed)

    def __len__(self):
        """Returns number of samples."""
        return self.n_samples

    def __iter__(self):
        return self

    def set_batch_size(self, new_batch_size):
        """Triggers a batch-size change."""
        self.batch_size = new_batch_size
        self.prepare_batches()

    def next(self):
        """Returns the next set of data from the iterator."""
        try:
            data = next(self._iter)
        except StopIteration as si:
            self.rewind()
            raise
        else:
            return OrderedDict([(k, data[i]) for i,k in enumerate(self._keys)])

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
