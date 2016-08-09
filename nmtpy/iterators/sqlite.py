#!/usr/bin/env python
from six.moves import range
from six.moves import zip

import sqlite3
import sys

import random
from collections import OrderedDict

import numpy as np

from ..nmtutils import sent_to_idx
from ..typedef  import INT, FLOAT
from .iterator import Iterator

class SQLIterator(object):
    """An efficient sqlite iterator for big parallel corpora."""
    def __init__(self, batch_size,
                 sql_file,
                 trg_dict=None, src_dict=None,
                 n_words_trg=0, n_words_src=0,
                 sort_by_length=False,
                 shuffle=False):

        self.n_samples = 0

        self.shuffle = shuffle
        if self.shuffle:
            # For minibatch shuffling
            random.seed(1234)

        self.sort_by_length = sort_by_length

        # This is the on-disk sqlite database
        self.sql_file = sql_file

        # Target word dictionary and short-list limit
        # This may not be available during validation
        self.trg_dict = trg_dict
        self.n_words_trg = n_words_trg

        # Source word dictionary and short-list limit
        # This may not be available if the task is image -> description
        self.src_dict = src_dict
        self.n_words_src = n_words_src

        # Batch size
        self.batch_size = batch_size

        # Whether to shuffle after each epoch
        self.shuffle = shuffle

        # keys define what to return during iteration
        self.__keys = ["x", "x_mask", "y", "y_mask"]

        # Read the data
        self.read()

    def __len__(self):
        return self.n_samples

    def set_batch_size(self, bs):
        """Sets the batch size and recreates batch idxs."""
        self.batch_size = bs

        if self.batch_size == 1:
            self.__keys = [k for k in self.__keys if not k.endswith("_mask")]

        # Create batch idxs
        self.__batches = [xrange(i, min(i+self.batch_size, self.n_samples)) \
                            for i in range(0, self.n_samples, self.batch_size)]
        self.__batch_iter = iter(self.__batches)

    def rewind(self):
        """Reshuffle if requested."""
        self.__batch_iter = iter(self.__batches)
        if self.shuffle:
            random.shuffle(self.samples)

    def __iter__(self):
        return self

    def prepare_batches(self):
        pass

    def read(self):
        # Load the samples
        conn = sqlite3.connect(self.sql_file)
        cur = conn.cursor()
        self.samples = cur.execute('SELECT src,trg FROM data').fetchall()
        con.close()
        self.n_samples = len(self.samples)

        # Some statistics
        unk_trg = 0
        unk_src = 0
        total_src_words = []
        total_trg_words = []

        # Let's map the sentences once to idx's
        for sample in self.samples:
            sample[0] = sent_to_idx(self.src_dict, sample[0], self.n_words_src)
            total_src_words.extend(sample[0])
            sample[1] = sent_to_idx(self.trg_dict, sample[1], self.n_words_trg)
            total_trg_words.extend(sample[0])

        self.unk_src = total_src_words.count(1)
        self.unk_trg = total_trg_words.count(1)
        self.total_src_words = len(total_src_words)
        self.total_trg_words = len(total_trg_words)

        self.set_batch_size(self.batch_size)

    def next(self):
        try:
            # Get batch idxs
            idxs = next(self.__batch_iter)
            batch = OrderedDict()

            batch['x'], batch['x_mask'] = Iterator.mask_data([self.samples[i][1] for i in idxs])
            batch['y'], batch['y_mask'] = Iterator.mask_data([self.samples[i][0] for i in idxs])

            return batch
        except StopIteration as si:
            self.rewind()
            raise
