#!/usr/bin/env python
from six.moves import range
from six.moves import zip

import numpy as np

from collections import OrderedDict

from ..sysutils import fopen
from .iterator  import Iterator
from .homogeneous import HomogeneousData

"""Parallel text iterator for translation data."""
class BiTextIterator(Iterator):
    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None, **kwargs):
        super(BiTextIterator, self).__init__(batch_size, seed, mask, shuffle_mode)

        assert 'srcfile' in kwargs, "Missing argument srcfile"
        assert 'trgfile' in kwargs, "Missing argument trgfile"
        assert 'srcdict' in kwargs, "Missing argument srcdict"
        assert 'trgdict' in kwargs, "Missing argument trgdict"

        self.srcfile = kwargs['srcfile']
        self.trgfile = kwargs['trgfile']
        self.srcdict = kwargs['srcdict']
        self.trgdict = kwargs['trgdict']

        self.n_words_src = kwargs.get('n_words_src', 0)
        self.n_words_trg = kwargs.get('n_words_trg', 0)

        self.src_name = kwargs.get('src_name', 'x')
        self.trg_name = kwargs.get('trg_name', 'y')

        self._keys = [self.src_name]
        if self.mask:
            self._keys.append("%s_mask" % self.src_name)

        self._keys.append(self.trg_name)
        if self.mask:
            self._keys.append("%s_mask" % self.trg_name)

    def read(self):
        seqs = []
        sf = fopen(self.srcfile, 'r')
        tf = fopen(self.trgfile, 'r')

        for idx, (sline, tline) in enumerate(zip(sf, tf)):
            sline = sline.strip()
            tline = tline.strip()

            # Exception if empty line found
            if sline == "" or tline == "":
                raise Exception("Empty line(s) detected in parallel corpora.")

            sseq = [self.srcdict.get(w, 1) for w in sline.split(' ')]
            tseq = [self.trgdict.get(w, 1) for w in tline.split(' ')]

            # if given limit vocabulary
            if self.n_words_src > 0:
                sseq = [w if w < self.n_words_src else 1 for w in sseq]

            # if given limit vocabulary
            if self.n_words_trg > 0:
                tseq = [w if w < self.n_words_trg else 1 for w in tseq]

            # Append sequences to the list
            seqs.append((sseq, tseq))
        
        sf.close()
        tf.close()

        # Save sequences
        self._seqs = seqs

        # Number of training samples
        self.n_samples = len(self._seqs)

        if self.shuffle_mode == 'trglen':
            # Homogeneous batches ordered by target sequence length
            self._iter = HomogeneousData(self._seqs, self.batch_size, 1)
        elif self.shuffle_mode == 'simple':
            # Simple shuffle
            self._idxs = np.random.permutation(self.n_samples)
            self.prepare_batches()
        else:
            # Ordered
            self._idxs = np.arange(self.n_samples)
            self.prepare_batches()

    def prepare_batches(self):
        self._minibatches = []

        for i in range(0, self.n_samples, self.batch_size):
            batch_idxs = self._idxs[i:i + self.batch_size]
            src, src_mask = Iterator.mask_data([self._seqs[i][0] for i in batch_idxs])
            trg, trg_mask = Iterator.mask_data([self._seqs[i][1] for i in batch_idxs])
            self._minibatches.append((src, src_mask, trg, trg_mask))

        self.rewind()

    def rewind(self):
        if self.shuffle_mode != 'trglen':
            self._iter = iter(self._minibatches)
