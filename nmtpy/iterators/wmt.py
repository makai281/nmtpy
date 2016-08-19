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
        assert 'imgfile' in kwargs, "Missing argument imgfile"
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
        self.mode = kwargs.get('mode', 'all')

        # pkl file which contains a list of samples
        self.pklfile = kwargs['pklfile']
        # Resnet-50 image features file
        self.imgfile = kwargs['imgfile']

        # Source word dictionary and short-list limit
        # This may not be available if the task is image -> description (Not implemented)
        self.srcdict = kwargs['srcdict']
        # This may not be available during validation
        self.trgdict = kwargs.get('trgdict', None)

        self._keys = [self.src_name]
        if self.mask:
            self._keys.append("%s_mask" % self.src_name)

        # We have images in the middle
        self._keys.append("%s_img" % self.src_name)

        # Target may not be available during validation
        if self.trgdict:
            self._keys.append(self.trg_name)
            if self.mask:
                self._keys.append("%s_mask" % self.trg_name)

    def read(self):
        # Load image features file
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
            sample[4] = sent_to_idx(self.src_dict, sample[4], self.n_words_src)
            total_src_words.extend(sample[4])
            if self.trg_avail:
                sample[5] = sent_to_idx(self.trg_dict, sample[5], self.n_words_trg)
                total_trg_words.extend(sample[5])

        self.unk_src = total_src_words.count(1)
        self.unk_trg = total_trg_words.count(1)
        self.total_src_words = len(total_src_words)
        self.total_trg_words = len(total_trg_words)

        if self.shuffle_mode == 'trglen':
            # Homogeneous batches ordered by target sequence length
            self._iter = HomogeneousData(self._seqs, self.batch_size, 5)
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

            # Source image features
            img_idxs = [self._seqs[i][2] for i in batch_idxs]

            # Do this 196 x bsize x 512
            x_img = self.img_feats[img_idxs].transpose(1, 0, 2)

            data_and_mask = Iterator.mask_data([self._seqs[i][4] for i in batch_idxs])
            data_and_mask += x_img
            if self.trg_avail:
                data_and_mask += Iterator.mask_data([self._seqs[i][5] for i in batch_idxs])

            # TODO: We should handle this in the model
            #if self.batch_size == 1:
               ## Drop middle axis
               #x_img = x_img.squeeze()

            self._minibatches.append(data_and_mask)

        self.rewind()

    def rewind(self):
        if self.shuffle_mode != 'trglen':
            self._iter = iter(self._minibatches)

#    def next(self):
        #try:
            ## Get batch idxs
            #idxs = next(self.__batch_iter)
            #x = x_img = x_mask = y = y_mask = None

            ## Target sentence
            #if self.trg_avail:
                #y, y_mask = Iterator.mask_data([self._seqs[i][5] for i in idxs])

            ## Optional source sentences
            #if self.src_avail:
                #x, x_mask = Iterator.mask_data([self._seqs[i][4] for i in idxs])

            ## Source image features
            #img_idxs = [self._seqs[i][2] for i in idxs]
            ## Do this 196 x bsize x 512
            #x_img = self.img_feats[img_idxs].transpose(1, 0, 2)
            #if self.batch_size == 1:
                ## Drop middle axis
                #x_img = x_img.squeeze()

            #return OrderedDict([(k, locals()[k]) for k in self.__keys if locals()[k] is not None])
        #except StopIteration as si:
            #self.rewind()
            #raise

if __name__ == '__main__':
    from nmtpy.nmtutils import load_dictionary
    trg_dict, _ = load_dictionary("/lium/buster1/caglayan/wmt16/data/text/task1.norm.lc.max50.ratio3.tok/train.norm.lc.tok.de.pkl")
    src_dict, _ = load_dictionary("/lium/buster1/caglayan/wmt16/data/text/task1.norm.lc.max50.ratio3.tok/train.norm.lc.tok.en.pkl")

    ite = WMTIterator(32,
                    "/lium/trad4a/wmt/2016/caglayan/data/task2/cross-product-min3-max50-minvocab5-train-680k/flickr_30k_align.train.pkl",
                    "/tmp/conv54_vgg_feats_hdf5-flickr30k.train.npy",
                    trg_dict, None)
    for i in range(2):
        print "Iterating..."
        for batch in ite:
            v = batch.keys()
            assert v[0] == "x_img"
            assert v[1] == "y"
            assert v[2] == "y_mask"

    ite = WMTIterator(32,
                    "/lium/trad4a/wmt/2016/caglayan/data/task2/cross-product-min3-max50-minvocab5-train-680k/flickr_30k_align.train.pkl",
                    "/tmp/conv54_vgg_feats_hdf5-flickr30k.train.npy",
                    trg_dict, src_dict)
    for i in range(2):
        print "Iterating..."
        for batch in ite:
            v = batch.keys()
            assert v[0] == "x"
            assert v[1] == "x_mask"
            assert v[2] == "x_img"
            assert v[3] == "y"
            assert v[4] == "y_mask"

    ite = WMTIterator(32,
                    "/lium/trad4a/wmt/2016/caglayan/data/task2/cross-product-min3-max50-minvocab5-train-680k/flickr_30k_align.train.pkl",
                    None,
                    trg_dict, src_dict)
    for i in range(2):
        print "Iterating..."
        for batch in ite:
            v = batch.keys()
            assert v[0] == "x"
            assert v[1] == "x_mask"
            assert v[2] == "y"
            assert v[3] == "y_mask"
