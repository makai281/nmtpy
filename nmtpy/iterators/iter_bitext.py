#!/usr/bin/env python
from six.moves import range
from six.moves import zip

import numpy as np

from collections import OrderedDict
from ..sysutils import fopen
from ..nmtutils import mask_data
from ..typedef  import INT, FLOAT

import random

"""Parallel text iterator for translation data."""
class BiTextIterator(object):
    def __init__(self, src_data, src_dict,
                       trg_data, trg_dict,
                       batch_size,
                       n_words_src=0, n_words_trg=0,
                       maxlen=50,
                       src_name='x', trg_name='y', n_splits=1,
                       maxlen_as_n_src_tsteps=False):

        # For minibatch shuffling
        random.seed(1234)

        # If n_splits > 1, it means that we have several sources
        # and several targets like in the crosslingual image description task.
        self.src_data = src_data
        self.src_dict = src_dict

        self.trg_data = trg_data
        self.trg_dict = trg_dict

        self.batch_size = batch_size

        self.n_words_src = n_words_src
        self.n_words_trg = n_words_trg

        self.src_name = src_name
        self.trg_name = trg_name

        self.n_splits = n_splits
        self.maxlen = maxlen

        # This is for fixed timesteps batches as input
        # Not used for recurrent networks
        self.n_src_tsteps = -1
        if maxlen_as_n_src_tsteps:
            self.n_src_tsteps = self.maxlen

        self.n_samples = 0
        self.__seqs = []
        self.__idxs = []
        self.__minibatches = []
        self.__return_keys = [self.src_name,
                              "%s_mask" % self.src_name,
                              self.trg_name,
                              "%s_mask" % self.trg_name]
        self.__iter = None

        self.read()

    def set_batch_size(self, bs):
        self.batch_size = bs

    def rewind(self):
        self.__iter = iter(self.__minibatches)

    def __iter__(self):
        return self

    def get_idxs(self):
        return self.__idxs

    def read(self):
        self.__max_filt = 0
        self.__seqs = []
        self.__idxs = []
        sf = fopen(self.src_data, 'r')
        tf = fopen(self.trg_data, 'r')

        for idx, (sline, tline) in enumerate(zip(sf, tf)):
            sline = sline.strip()
            tline = tline.strip()

            # Exception if empty line found
            if sline == "" or tline == "":
                raise Exception("Empty line(s) detected in parallel corpora.")

            sline = sline.split(" ")
            tline = tline.split(" ")

            # Filter out long sentences
            if self.maxlen > 0 and len(sline) > self.maxlen or len(tline) > self.maxlen:
                self.__max_filt += 1
                continue

            sseq = [self.src_dict.get(w, 1) for w in sline]
            tseq = [self.trg_dict.get(w, 1) for w in tline]

            # if given limit vocabulary
            if self.n_words_src > 0:
                sseq = [w if w < self.n_words_src else 1 for w in sseq]

            # if given limit vocabulary
            if self.n_words_trg > 0:
                tseq = [w if w < self.n_words_trg else 1 for w in tseq]

            # Append sequences to the list
            self.__seqs.append((sseq, tseq))
            self.__idxs += [idx]
        
        # Save sentence count
        self.n_samples = len(self.__idxs)

        sf.close()
        tf.close()

    def __repr__(self):
        s = self.__class__.__name__
        if self.n_samples > 0:
            s += " of %d parallel sentences" % self.n_samples
            if self.__max_filt > 0:
                s += "%d sentences filtered out as long." % self.__max_filt
        return s

    def prepare_batches(self, shuffle=False, sort=False):
        sample_idxs = np.arange(self.n_samples)
        self.__minibatches = []

        if sort:
            # Sort samples by target sentence length
            sample_idxs = sorted(sample_idxs, key=lambda i: len(self.__seqs[i][1]))
        elif shuffle:
            # Shuffle samples
            np.random.shuffle(sample_idxs)

        for i in range(0, self.n_samples, self.batch_size):
            batch_idxs = sample_idxs[i:i + self.batch_size]
            x, x_mask = mask_data([self.__seqs[i][0] for i in batch_idxs], self.n_src_tsteps)
            y, y_mask = mask_data([self.__seqs[i][1] for i in batch_idxs])
            self.__minibatches.append((batch_idxs, x, x_mask, y, y_mask))

        if sort and shuffle:
            # The last one is probably smaller than batch_size, exclude it
            all_but_last = self.__minibatches[:-1]
            random.shuffle(all_but_last)
            # Add the last one now
            all_but_last.append(self.__minibatches[-1])
            self.__minibatches = all_but_last

            # Recreate sample idxs from shuffled batches
            sample_idxs = []
            for batch in self.__minibatches:
                sample_idxs.extend(batch[0])

        self.__iter = iter(self.__minibatches)
        self.__idxs = sample_idxs

    def next(self):
        try:
            data = next(self.__iter)
        except StopIteration as si:
            self.rewind()
            raise
        except AttributeError as ae:
            raise Exception("You need to call prepare_batches() first.")
        else:
            return OrderedDict([(k,data[i+1]) for i,k in enumerate(self.__return_keys)])
