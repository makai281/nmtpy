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
                       src_name='x', trg_name='y'):

        # For minibatch shuffling
        random.seed(1234)

        self.src_data = src_data
        self.src_dict = src_dict

        self.trg_data = trg_data
        self.trg_dict = trg_dict

        self.batch_size = batch_size

        self.n_words_src = n_words_src
        self.n_words_trg = n_words_trg

        self.src_name = src_name
        self.trg_name = trg_name

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

    def __repr__(self):
        return "src: %s, trg: %s" % (self.src_data, self.trg_data)

    def set_batch_size(self, bs):
        self.batch_size = bs
        self.prepare_batches()

    def rewind(self):
        self.__iter = iter(self.__minibatches)

    def __iter__(self):
        return self

    def get_idxs(self):
        return self.__idxs

    def read(self):
        #self.__max_filt = 0
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

    def prepare_batches(self):
        sample_idxs = np.arange(self.n_samples)
        self.__minibatches = []

        for i in range(0, self.n_samples, self.batch_size):
            batch_idxs = sample_idxs[i:i + self.batch_size]
            x, x_mask = mask_data([self.__seqs[i][0] for i in batch_idxs])
            y, y_mask = mask_data([self.__seqs[i][1] for i in batch_idxs])
            self.__minibatches.append((batch_idxs, x, x_mask, y, y_mask))

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

### Test
if __name__ == '__main__':
    import os
    src = "~/wmt16/data/text/norm.moses.tok/train.norm.lc.tok.en"
    trg = "~/wmt16/data/text/norm.moses.tok/train.norm.lc.tok.de"
    sdc = "~/wmt16/data/text/norm.moses.tok/train.norm.lc.tok.en.pkl"
    tdc = "~/wmt16/data/text/norm.moses.tok/train.norm.lc.tok.de.pkl"

    from ..nmtutils import load_dictionary,idx_to_sent
    src_dict, src_idict = load_dictionary(os.path.expanduser(sdc))
    trg_dict, trg_idict = load_dictionary(os.path.expanduser(tdc))

    iterator = BiTextIterator(os.path.expanduser(src), src_dict,
                              os.path.expanduser(trg), trg_dict, batch_size=32)
    iterator.prepare_batches()
    idxs1 = iterator.get_idxs()

    d = {}
    for batch in iterator:
        for s,t in zip(batch['x'].T, batch['y'].T):
            src_sent = idx_to_sent(src_idict, s)
            trg_sent = idx_to_sent(trg_idict, t)
            if src_sent in d:
                print "Key already available: %s" % src_sent
            else:
                d[src_sent] = trg_sent
