#!/usr/bin/env python
from six.moves import range
from six.moves import zip

import random
import cPickle
from collections import OrderedDict

import numpy as np

from ..nmtutils import mask_data, idx_to_sent, load_dictionary
from ..typedef  import INT, FLOAT

class WMT16Iterator(object):
    def __init__(self, pkl_file, pkl_splits, batch_size,
                 trg_dict, src_dict=None, src_img=True,
                 n_words_trg=0, n_words_src=0,
                 shuffle=False):

        self.n_samples = 0
        self.__minibatches = []
        self.__iter = None
        self.__seqs = []
        self.__return_keys = []

        # For minibatch shuffling
        random.seed(1234)

        # Input images may be optional as well
        self.src_img = src_img

        # PKL file
        self.pkl_file = pkl_file

        # split can be "train" or "train,valid", etc.
        self.splits = pkl_splits.split(",")

        # Target word dictionary and short-list limit
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

        # src_name can be multiple like 'x_img', 'x',  trg_name is 'y'
        if self.src_dict:
            self.__return_keys.extend(["x", "x_mask"])

        if self.src_img:
            # Image features and target sentences
            self.__return_keys.append("x_img")

        # target sentences should always be available    
        self.__return_keys.extend(["y", "y_mask"])

        # Read the data
        self.read()

    def __repr__(self):
        return "%s (splits: %s)" % (self.pkl_file, self.splits)

    def set_batch_size(self, bs):
        """Sets the batch size and reorganizes minibatches."""
        self.batch_size = bs
        self.prepare_batches()

    def rewind(self):
        """Rewinds the iterator and shuffles if requested."""
        if self.shuffle:
            random.shuffle(self.__minibatches)

        self.__iter = iter(self.__minibatches)

    def __iter__(self):
        return self

    def __sent_to_idx(vocab, tokens, limit):
        idxs = []
        for w in tokens:
            # Get token, 1 if not available
            widx = vocab.get(w, 1)
            if limit > 0:
                widx = widx if widx < limit else 1
            idxs.append(widx)
        return idxs

    def read(self):
        """Reads the .pkl file and fills in the informations."""
        ##############
        with open(self.pkl_file) as f:
            d = cPickle.load(f)

        # These are the image features
        self.feats = d['feats']

        # 4096 for VGG FC, 2048 for ResNet FC
        # For conv this would be 512*14*14 = 100352
        self.img_dim = self.feats.shape[1]

        # Add ability to read multiple splits which will
        # help during final training on both train and valid
        for spl in self.splits:
            for src, trg, imgid, imgfilename in d['sents'][spl]:
                # We keep imgid's if requested
                seq = {'x_img' : imgid if self.src_img else None}
                # We always have target sentences
                seq['y'] = self.__sent_to_idx(self.trg_dict, trg, self.n_words_trg)
                # We put None's if the caller didn't request source sentences
                seq['x'] = self.__sent_to_idx(self.src_dict, src, self.n_words_src) if self.src_dict else None
                # Append it to the sequences
                self.__seqs.append(seq)

        # Save sentence count
        self.n_samples = len(self.__seqs)

    def prepare_batches(self):
        if self.shuffle:
            # Shuffle data
            random.shuffle(self.__seqs)

        self.__minibatches = []

        # Batch idxs
        batches = [xrange(i, min(i+self.batch_size, self.n_samples)) \
                for i in range(0, self.n_samples, self.batch_size)]

        for idxs in batches:
            if self.src_dict:
                x, x_mask = mask_data([self.__seqs[i]['x'] for i in idxs])
                self.__minibatches.extend([x, x_mask])

            # Source image features
            if self.src_img:
                self.__minibatches.append(np.vstack(self.feats[self.__seqs[i]['x_img']] for i in idxs))

            # Target sentence (always available)
            y, y_mask = mask_data([self.__seqs[i]['y'] for i in idxs])
            self.__minibatches.extend([y, y_mask])


        self.__iter = iter(self.__minibatches)

    def next(self):
        try:
            data = next(self.__iter)
        except StopIteration as si:
            self.rewind()
            raise
        except AttributeError as ae:
            raise Exception("You need to call prepare_batches() first.")
        else:
            return OrderedDict([(k, data[k]) for k in self.__return_keys])
