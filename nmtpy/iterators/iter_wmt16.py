#!/usr/bin/env python
from six.moves import range
from six.moves import zip

import random
from collections import OrderedDict

import numpy as np

from nmtpy.nmtutils import mask_data, idx_to_sent, load_dictionary
from nmtpy.typedef  import INT, FLOAT

class WMT16Iterator(object):
    def __init__(self, npz_file, splits, batch_size,
                 trg_dict, src_dict=None, src_img=True,
                 n_words_trg=0, n_words_src=0,
                 shuffle=False):

        self.n_samples = 0
        self.__seqs = []
        self.__keys = []

        # For minibatch shuffling
        random.seed(1234)

        # Input images may be optional as well
        self.src_img = src_img

        # npz file
        self.npz_file = npz_file

        # split can be "train" or "train,valid", etc.
        self.splits = splits.split(",")

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
            self.__keys.extend(["x", "x_mask"])

        if self.src_img:
            # Image features and target sentences
            self.__keys.append("x_img")

        # target sentences should always be available    
        self.__keys.extend(["y", "y_mask"])

        # Read the data
        self.read()

    def __repr__(self):
        return self.npz_file

    def set_batch_size(self, bs):
        """Sets the batch size and recreates batch idxs."""
        self.batch_size = bs

        # Create batch idxs
        self.__batches = [xrange(i, min(i+self.batch_size, self.n_samples)) \
                            for i in range(0, self.n_samples, self.batch_size)]

    def rewind(self):
        """Reshuffle if requested."""
        if self.shuffle:
            random.shuffle(self.__seqs)

    def __iter__(self):
        return self

    def __sent_to_idx(self, vocab, tokens, limit):
        idxs = []
        for w in tokens:
            # Get token, 1 if not available
            widx = vocab.get(w, 1)
            if limit > 0:
                widx = widx if widx < limit else 1
            idxs.append(widx)
        return idxs

    def read(self):
        """Reads the .npz file and fills in the informations."""
        ##############
        f = np.load(self.npz_file)

        # These are the image features
        self.feats = f['feats']
        d = {}

        # Load splits
        for sp in ("train", "valid", "test"):
            if sp in f:
                d[sp] = f[sp].tolist()

        f.close()

        # Let the reshaping work to the model's load_data() file
        self.img_dim = self.feats.shape[1]

        # Read training samples
        for src, trg, imgid, imgfilename in d['train']:
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

        # Shuffle sequences. This doesn't harm anything as
        # imgid's are also in the sequences.
        if self.shuffle:
            random.shuffle(self.__seqs)

        # Create batch idxs
        self.set_batch_size(self.batch_size)

    def next(self):
        try:
            # Get batch idxs
            idxs = next(self.__batches)
            if self.src_dict:
                x, x_mask = mask_data([self.__seqs[i]['x'] for i in idxs])
                self.__minibatches.extend([x, x_mask])

            # Source image features
            if self.src_img:
                self.__minibatches.append(np.vstack(self.feats[self.__seqs[i]['x_img']] for i in idxs))

            # Target sentence (always available)
            y, y_mask = mask_data([self.__seqs[i]['y'] for i in idxs])
            self.__minibatches.extend([y, y_mask])


        except StopIteration as si:
            self.rewind()
            raise
        else:
            return OrderedDict([(k, data[k]) for k in self.__keys])
