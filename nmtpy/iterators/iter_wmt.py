#!/usr/bin/env python
from six.moves import range
from six.moves import zip

import cPickle

import random
from collections import OrderedDict

import numpy as np

from ..nmtutils import mask_data, sent_to_idx
from ..typedef  import INT, FLOAT

class WMTIterator(object):
    def __init__(self, batch_size,
                 pkl_file,
                 img_feats_file=None,
                 trg_dict=None, src_dict=None,
                 n_words_trg=0, n_words_src=0,
                 shuffle=False):

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

        # Whether to shuffle after each epoch
        self.shuffle = shuffle

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

    def set_batch_size(self, bs):
        """Sets the batch size and recreates batch idxs."""
        self.batch_size = bs

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
        if self.img_feats_file:
            self.img_feats = np.load(self.img_feats_file)

            # Transpose and fix dimensionality of convolutional patches
            self.img_feats.shape = (self.img_feats.shape[0], 512, 14, 14)
            self.img_feats.shape = (self.img_feats.shape[0], 512, 196)
            self.img_feats = self.img_feats.transpose(0, 2, 1)

        # Load the samples
        with open(self.pkl_file, 'rb') as f:
            # This import needs to be here so that unpickling works correctly
            from ..typedef import Sample
            self.samples = cPickle.load(f)

        # Check for what is available
        ss = self.samples[0]
        if ss.swords is not None and self.src_dict:
            self.src_avail = True
        if ss.twords is not None and self.trg_dict:
            self.trg_avail = True
        if ss.imgid is not None and self.img_feats is not None:
            self.img_avail = True

        # We now have a list of Sample()'s
        self.n_samples = len(self.samples)

        seqs = []

        # Let's map the sentences once to idx's
        for sample in self.samples:
            sample = list(sample)
            if self.src_avail:
                sample[4] = sent_to_idx(self.src_dict, sample[4], self.n_words_src)
            if self.trg_avail:
                sample[5] = sent_to_idx(self.trg_dict, sample[5], self.n_words_trg)
            seqs.append(sample)
        self.samples = seqs

        self.set_batch_size(self.batch_size)

    def next(self):
        try:
            # Get batch idxs
            idxs = next(self.__batch_iter)
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
                x_img = self.img_feats[img_idxs]

            return OrderedDict([(k, locals()[k]) for k in self.__keys if locals()[k] is not None])
        except StopIteration as si:
            self.rewind()
            raise

if __name__ == '__main__':
    from nmtpy.nmtutils import load_dictionary
    trg_dict, _ = load_dictionary("/lium/buster1/caglayan/wmt16/data/text/task1.norm.lc.max50.ratio3.tok/train.norm.lc.tok.de.pkl")
    src_dict, _ = load_dictionary("/lium/buster1/caglayan/wmt16/data/text/task1.norm.lc.max50.ratio3.tok/train.norm.lc.tok.en.pkl")
    from ..typedef import Sample

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
