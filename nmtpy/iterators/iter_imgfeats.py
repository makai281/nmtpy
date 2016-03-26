#!/usr/bin/env python
from six.moves import range
from six.moves import zip

import numpy as np

from collections import OrderedDict
from ..typedef import INT, FLOAT

class ImageFeatsIterator(object):
    def __init__(self, filename, batch_size, norm=False,
                 idxs=None, do_mask=False, n_timesteps=None, filetype="npy",
                 hdf5_node='feats', data_name="x_img"):
        # filetype: hdf5, npy
        # beware that hdf5 is really slow on huge files
        # like the 11GB ConvNet layer of VGG for Flickr30k.
        # Converting that features to .npy and putting
        # the file in /tmp makes the loading time ~5 secs.
        #
        self.__data = None

        if filetype == "npy":
            self.__data = np.load(filename)
        elif filetype == "hdf5":
            import h5py
            h = h5py.File(filename, 'r')
            node = h[hdf5_node]
            self.__data = np.empty(node.shape, node.dtype)
            node.read_direct(self.__data)
            h.close()
        else:
            raise Exception("File format not recognized %s" % filename)

        self.__data = self.__data.astype(FLOAT)

        if idxs is not None:
            # Receive iteration order from the caller. Useful
            # for shuffled text datasets or sub-splits.
            self.__idxs = np.array(idxs, dtype=np.int32)
        else:
            # Ordered iteration
            self.__idxs = np.arange(self.__data.shape[0])

        self.n_timesteps = n_timesteps
        if self.n_timesteps:
            # If this is 196..
            self.n_timesteps = n_timesteps
            # This one will be 512 for conv features from VGG
            self.n_feats = self.__data.shape[1] / n_timesteps

        # Number of samples
        self.n_samples = self.__idxs.size

        self.batch_size = batch_size
        self.do_mask = do_mask
        self.data_name = data_name
        self.norm = norm
        if self.norm:
            self.normalize()

        self.dim = self.__data.shape[1]
        self.rewind()

    def set_batch_size(self, bs):
        self.batch_size = bs

    def prepare_batches(self):
        pass

    def normalize(self):
        eps = 1e-5
        x_remove_mu = self.__data - self.__data.mean(axis=0)
        s2 = np.sum((x_remove_mu**2), axis=0).mean(axis=0)
        self.__data = x_remove_mu / np.sqrt(s2 + eps)

    def __iter__(self):
        return self

    def next(self):
        if self.__next == self.n_samples:
            self.rewind()
            raise StopIteration

        # n_samples x n_feats
        d = self.__data[self.__idxs[self.__next:self.__next + self.batch_size]]
        # Increment __next to point to the next batch's first sample
        self.__next += d.shape[0]

        # Change dimensions to fit the model
        if self.n_timesteps:
            # e.g. batch_size x 196 x 512
            d.shape = (d.shape[0], self.n_timesteps, self.n_feats)
            d = np.swapaxes(d, 0, 1)
            # Now we have, n_timesteps x n_samples x n_feats

        od = OrderedDict([(self.data_name, d)])
        if self.do_mask:
            od['%s_mask' % self.data_name] = np.ones_like(d)

        return od

    def rewind(self):
        self.__next = 0

##### TEST
def test_image_iterator():
    data_file = "/lium/trad4a/wmt/2016/data/vgg-feats/npy/fc7_vgg_feats_hdf5-flickr30k.valid.npy"
    orig_data = np.load(data_file)

    sorted_idxs = np.arange(orig_data.shape[0])
    subset_idxs = sorted_idxs[:768]
    shuffled_idxs = np.arange(orig_data.shape[0])
    np.random.shuffle(shuffled_idxs)
    shuffled_data = orig_data[shuffled_idxs]

    orig_data_5 = orig_data[:]
    orig_data_5.shape = (orig_data_5.shape[0], 256, -1)
    orig_data_5 = np.swapaxes(orig_data_5, 0, 1)

    for bs in [1, 2, 12, 32, 63, 507, 1014]:
        print "Batch size: %d" % bs
        a = None
        it = ImageFeatsIterator(data_file, batch_size=bs, idxs=None, do_mask=False)
        for batch in it:
            img = batch[it.data_name]
            if a is None:
                a = img
            else:
                a = np.vstack([a, img])

        assert np.allclose(orig_data, a), "idxs=None, bs=%d, FAIL" % bs
        print "1st test OK"

        a = None
        it = ImageFeatsIterator(data_file, batch_size=bs, idxs=subset_idxs, do_mask=True)
        for batch in it:
            img = batch[it.data_name]
            if a is None:
                a = img
            else:
                a = np.vstack([a, img])

        assert np.allclose(orig_data[:768], a), "idxs=subset, bs=%d, FAIL" % bs
        print "2nd test OK"

        a = None
        it = ImageFeatsIterator(data_file, batch_size=bs, idxs=sorted_idxs, do_mask=True)
        for batch in it:
            img = batch[it.data_name]
            if a is None:
                a = img
            else:
                a = np.vstack([a, img])

        assert np.allclose(orig_data, a), "idxs=sorted, bs=%d, FAIL" % bs
        print "3rd test OK"

        a = None
        it = ImageFeatsIterator(data_file, batch_size=bs, idxs=shuffled_idxs, do_mask=True)
        for batch in it:
            img = batch[it.data_name]
            if a is None:
                a = img
            else:
                a = np.vstack([a, img])

        assert np.allclose(shuffled_data, a), "idxs=sorted, bs=%d, FAIL" % bs   
        print "4th test OK"

        a = None
        it = ImageFeatsIterator(data_file, batch_size=bs, idxs=None, do_mask=True, n_timesteps=256)
        for batch in it:
            img = batch[it.data_name]
            if a is None:
                a = img
            else:
                a = np.concatenate([a, img], axis=1)

        assert np.allclose(orig_data_5, a), "idxs=sorted, bs=%d, FAIL" % bs
        print "5th test OK"


if __name__ == '__main__':
    test_image_iterator()
