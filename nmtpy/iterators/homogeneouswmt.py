#!/usr/bin/env python
from six.moves import range
from six.moves import zip

import cPickle
import sys

import random
from collections import OrderedDict

import numpy as np

from nmtpy.nmtutils import sent_to_idx
from nmtpy.typedef  import INT, FLOAT
from homogeneous_data import HomogeneousData

def mask_data(seqs):
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)

    # For ff-enc, we need fixed tsteps in the input
    maxlen = np.max(lengths) + 1

    # Shape is (t_steps, samples)
    x = np.zeros((maxlen, n_samples)).astype(INT)
    x_mask = np.zeros_like(x).astype(FLOAT)

    for idx, s_x in enumerate(seqs):
        x[:lengths[idx], idx] = s_x
        x_mask[:lengths[idx] + 1, idx] = 1.

    return x, x_mask

# Each element of the list that is pickled is in the following format:
# [ssplit, tsplit, imgid, imgname, swords, twords]

### NOTE
### This is not ready for production use and is not tested!

class WMTHomogeneousIterator(object):
    def __init__(self, batch_size,
                 pkl_file,
                 img_feats_file=None,
                 trg_dict=None, src_dict=None,
                 n_words_trg=0, n_words_src=0,
                 mode='all'):

        self.n_samples = 0

        # These are set after reading the pkl file
        self.src_avail = False
        self.trg_avail = False
        self.img_avail = False

        # For minibatch shuffling
        random.seed(1234)

        # pkl file which contains a list of Sample objects
        self.pkl_file = pkl_file

        # This is expected to be a .npy file
        self.img_feats_file = img_feats_file
        self.img_feats = None

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

        # 'all'     : Use everything available in the pkl file (default)
        # 'single'  : Take only the first pair e.g., train0.en->train0.de
        # 'pairs'   : Take only one-to-one pairs e.g., train_i.en->train_i.de
        self.mode = mode

        # keys define what to return during iteration
        self.__keys = []
        if self.src_dict:
            # We have source sentences
            self.__keys.extend(["x", "x_mask"])

        if self.img_feats_file:
            # We have source images
            self.__keys.append("x_img")

        if self.trg_dict:
            # We have target sentences
            self.__keys.extend(["y", "y_mask"])

       # Read the data
        self.read()

    def __len__(self):
        return self.n_samples

    def set_batch_size(self, bs):
        """Sets the batch size and recreates batch idxs."""
        self.batch_size = bs

        if self.batch_size == 1:
            self.__keys = [k for k in self.__keys if not k.endswith("_mask")]

    def rewind(self):
        """Reshuffle if requested."""
        pass

    def __iter__(self):
        return self

    def prepare_batches(self):
        pass

    def read(self):
        if self.img_feats_file:
            self.img_feats = np.load(self.img_feats_file)

            # NOTE: Hacky check to distinguish btw resnet and VGG
            if self.img_feats.ndim == 2:
                # Transpose and fix dimensionality of convolutional patches
                # for VGG
                self.img_feats.shape = (self.img_feats.shape[0], 512, 14, 14)
                self.img_feats.shape = (self.img_feats.shape[0], 512, 196)
                self.img_feats = self.img_feats.transpose(0, 2, 1)

        # Load the samples
        with open(self.pkl_file, 'rb') as f:
            # This import needs to be here so that unpickling works correctly
            self.samples = cPickle.load(f)

        # Check for what is available
        ss = self.samples[0]
        if ss[0] is not None and self.src_dict:
            self.src_avail = True
        if ss[1] is not None and self.trg_dict:
            self.trg_avail = True
        if ss[2] is not None and self.img_feats is not None:
            self.img_avail = True

        if self.mode == 'single':
            # Just take the first src-trg pair. Useful for validation
            if ss[0] is not None and ss[1] is not None:
                self.samples = [s for s in self.samples if (s[0] == s[1] == 0)]
            elif self.src_avail:
                self.samples = [s for s in self.samples if (s[0] == 0)]
        elif ss[1] is not None and self.mode == 'pairs':
            # Take the pairs with split idx's equal
            self.samples = [s for s in self.samples if s[0] == s[1]]

        # We now have a list of samples
        self.n_samples = len(self.samples)

        # Some statistics
        unk_trg = 0
        unk_src = 0
        total_src_words = []
        total_trg_words = []

        # Let's map the sentences once to idx's
        if self.src_avail or self.trg_avail:
            for sample in self.samples:
                if self.src_avail:
                    sample[4] = sent_to_idx(self.src_dict, sample[4], self.n_words_src)
                    total_src_words.extend(sample[4])
                if self.trg_avail:
                    sample[5] = sent_to_idx(self.trg_dict, sample[5], self.n_words_trg)
                    total_trg_words.extend(sample[5])

        self.unk_src = total_src_words.count(1)
        self.unk_trg = total_trg_words.count(1)
        self.total_src_words = len(total_src_words)
        self.total_trg_words = len(total_trg_words)

        self.__base_iter = HomogeneousData(data=self.samples,
                                           batch_size=self.batch_size,
                                           target_func=lambda s: s[5])

        self.set_batch_size(self.batch_size)

    def next(self):
        try:
            # Get batch idxs
            idxs = next(self.__base_iter)

            x = x_img = x_mask = y = y_mask = None

            # Target sentence
            if self.trg_avail:
                y, y_mask = mask_data([self.samples[i][5] for i in idxs])

            # Optional source sentences
            if self.src_avail:
                x, x_mask = mask_data([self.samples[i][4] for i in idxs])

            # Source image features
            if self.img_avail:
                img_idxs = [self.samples[i][2] for i in idxs]
                # Do this 196 x bsize x conv_dim
                x_img = self.img_feats[img_idxs].transpose(1, 0, 2)
                if self.batch_size == 1:
                    # Drop middle axis
                    x_img = x_img.squeeze()

            return OrderedDict([(k, locals()[k]) for k in self.__keys if locals()[k] is not None])
        except StopIteration as si:
            self.rewind()
            raise

if __name__ == '__main__':
    from nmtpy.nmtutils import load_dictionary
    trg_dict, _ = load_dictionary("/lium/buster1/caglayan/wmt16/data/text/task1.norm.lc.max50.ratio3.tok/train.norm.lc.tok.de.pkl")
    src_dict, _ = load_dictionary("/lium/buster1/caglayan/wmt16/data/text/task1.norm.lc.max50.ratio3.tok/train.norm.lc.tok.en.pkl")

    ite = WMTHomogeneousIterator(32,
                    "/lium/trad4a/wmt/2016/caglayan/data/task2/cross-product-min3-max50-minvocab5-train-680k/flickr_30k_align.train.pkl",
                    "/tmp/conv54_vgg_feats_hdf5-flickr30k.train.npy",
                    trg_dict=trg_dict, src_dict=src_dict, mode='pairs')

    n_src = 0
    n_trg = 0
    c = 0
    for batch in ite:
        c += 1
        n_src += batch['x_mask'].sum()
        n_trg += batch['y_mask'].sum()

    print n_src
    print n_trg
    print 'loop count: %d' % c

