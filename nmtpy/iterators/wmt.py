#!/usr/bin/env python
from six.moves import range
from six.moves import zip

import cPickle
import sys

import random
from collections import OrderedDict

import numpy as np

from ..nmtutils import sent_to_idx
from ..typedef  import INT, FLOAT
from .iterator import Iterator
from .homogeneous import HomogeneousData

# This is an iterator specifically to be used by the .pkl
# corpora files created for WMT16 Shared Task on Multimodal Machine Translation
# Each element of the list that is pickled is in the following format:
# [src_split_idx, trg_split_idx, imgid, imgname, src_words, trg_words]

class WMTIterator(Iterator):
    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None, **kwargs):
        super(WMTIterator, self).__init__(batch_size, seed, mask, shuffle_mode)

        assert 'pklfile' in kwargs, "Missing argument pklfile"
        assert 'srcdict' in kwargs, "Missing argument srcdict"

        # Short-list sizes
        self.n_words_src = kwargs.get('n_words_src', 0)
        self.n_words_trg = kwargs.get('n_words_trg', 0)

        # How do we refer to symbolic data variables?
        self.src_name = kwargs.get('src_name', 'x')
        self.trg_name = kwargs.get('trg_name', 'y')

        # How do we use the multimodal data?
        # 'all'     : All combinations (~725K parallel)
        # 'single'  : Take only the first pair e.g., train0.en->train0.de (~29K parallel)
        # 'pairs'   : Take only one-to-one pairs e.g., train_i.en->train_i.de (~145K parallel)
        self.mode = kwargs.get('mode', 'pairs')

        # pkl file which contains a list of samples
        self.pklfile = kwargs['pklfile']
        # Resnet-50 image features file
        self.imgfile = kwargs.get('imgfile', None)
        self.img_avail = self.imgfile is not None

        # Source word dictionary and short-list limit
        # This may not be available if the task is image -> description (Not implemented)
        self.srcdict = kwargs['srcdict']
        # This may not be available during validation
        self.trgdict = kwargs.get('trgdict', None)

        self._keys = [self.src_name]
        if self.mask:
            self._keys.append("%s_mask" % self.src_name)

        # We have images in the middle
        if self.imgfile:
            self._keys.append("%s_img" % self.src_name)

        # Target may not be available during validation
        if self.trgdict:
            self._keys.append(self.trg_name)
            if self.mask:
                self._keys.append("%s_mask" % self.trg_name)

    def read(self):
        # Load image features file if any
        if self.img_avail:
            self.img_feats = np.load(self.imgfile)

        # Load the corpora
        with open(self.pklfile, 'rb') as f:
            self._seqs = cPickle.load(f)

        # Check for what is available
        ss = self._seqs[0]
        if ss[1] is not None and self.trgdict:
            self.trg_avail = True

        if self.mode == 'single':
            # Just take the first src-trg pair. Useful for validation
            if ss[1] is not None:
                self._seqs = [s for s in self._seqs if (s[0] == s[1] == 0)]
            else:
                self._seqs = [s for s in self._seqs if (s[0] == 0)]

        elif ss[1] is not None and self.mode == 'pairs':
            # Take the pairs with split idx's equal
            self._seqs = [s for s in self._seqs if s[0] == s[1]]

        # We now have a list of samples
        self.n_samples = len(self._seqs)

        # Some statistics
        unk_trg = 0
        unk_src = 0
        total_src_words = []
        total_trg_words = []

        # Let's map the sentences once to idx's
        for sample in self._seqs:
            sample[4] = sent_to_idx(self.srcdict, sample[4], self.n_words_src)
            total_src_words.extend(sample[4])
            if self.trg_avail:
                sample[5] = sent_to_idx(self.trgdict, sample[5], self.n_words_trg)
                total_trg_words.extend(sample[5])

        self.unk_src = total_src_words.count(1)
        self.unk_trg = total_trg_words.count(1)
        self.total_src_words = len(total_src_words)
        self.total_trg_words = len(total_trg_words)

        self._process_batch = (lambda idxs: self.mask_seqs(idxs))

        if self.shuffle_mode == 'trglen':
            # Homogeneous batches ordered by target sequence length
            # Get an iterator over sample idxs
            self._iter = HomogeneousData(self._seqs, self.batch_size, trg_pos=5)
        else:
            self.rewind()

    def mask_seqs(self, idxs):
        """Prepares a list of padded tensors with their masks for the given sample idxs."""
        data_and_mask = list(Iterator.mask_data([self._seqs[i][4] for i in idxs]))

        # Source image features
        if self.img_avail:
            img_idxs = [self._seqs[i][2] for i in idxs]

            # Do this 196 x bsize x 512
            x_img = self.img_feats[img_idxs].transpose(1, 0, 2)

            # TODO: We should handle this in the model?
            x_img = x_img.squeeze() if self.batch_size == 1 else x_img

            data_and_mask += [x_img]

        if self.trg_avail:
            data_and_mask += list(Iterator.mask_data([self._seqs[i][5] for i in idxs]))

        return data_and_mask

    def prepare_batches(self):
        pass

    def rewind(self):
        if self.shuffle_mode != 'trglen':
            # Fill in the _idxs list for sample order
            if self.shuffle_mode == 'simple':
                # Simple shuffle
                self._idxs = np.random.permutation(self.n_samples).tolist()
            elif self.shuffle_mode is None:
                # Ordered
                self._idxs = np.arange(self.n_samples).tolist()
            self._iter = []
            for i in range(0, self.n_samples, self.batch_size):
                self._iter.append(self._idxs[i:i + self.batch_size])
            self._iter = iter(self._iter)

if __name__ == '__main__':
    from nmtpy.nmtutils import load_dictionary
    trg_dict, _ = load_dictionary("/lium/buster1/caglayan/wmt16/data/text/task1.norm.lc.max50.ratio3.tok/train.norm.lc.tok.de.pkl")
    src_dict, _ = load_dictionary("/lium/buster1/caglayan/wmt16/data/text/task1.norm.lc.max50.ratio3.tok/train.norm.lc.tok.en.pkl")

    ite = WMTIterator(batch_size=32,
                      pklfile="/lium/trad4a/wmt/2016/caglayan/data/task2/cross-product-min3-max50-minvocab5-train-680k/flickr_30k_align.train.pkl",
                      imgfile="/lium/trad4a/wmt/2016/data/resnet-feats/flickr30k_ResNets50_blck4_train.npy",
                      trgdict=trg_dict, srcdict=src_dict, mode='pairs', shuffle_mode='trglen')
    ite.read()
    for i in range(2):
        print "Iterating...", i
        for batch in ite:
            v = batch.keys()
            assert v[0] == "x"
            assert v[1] == "x_mask"
            assert v[2] == "x_img"
            assert v[3] == "y"
            assert v[4] == "y_mask"

    ite = WMTIterator(batch_size=32,
                      pklfile="/lium/trad4a/wmt/2016/caglayan/data/task2/cross-product-min3-max50-minvocab5-train-680k/flickr_30k_align.train.pkl",
                      trgdict=trg_dict, srcdict=src_dict, mode='pairs')
    ite.read()
    for i in range(2):
        print "Iterating...", i
        for batch in ite:
            v = batch.keys()
            assert v[0] == "x"
            assert v[1] == "x_mask"
            assert v[2] == "y"
            assert v[3] == "y_mask"
