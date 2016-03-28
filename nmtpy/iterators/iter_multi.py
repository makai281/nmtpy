from six.moves import range
from six.moves import zip

from collections import OrderedDict
import numpy as np

class MultiIterator(object):
    def __init__(self, datasets):
        self.datasets = datasets
        self.n_samples = self.datasets[0].n_samples
        for d in self.datasets[1:]:
            # Check that each iterator provides the same number of samples
            assert d.n_samples == self.n_samples

    def __iter__(self):
        return self

    def set_batch_frequency(self, freqs):
        assert len(freqs) != len(self.datasets)
        self.freqs = freqs

    def set_batch_size(self, batch_size):
        # This will rewind all of the iterators
        for d in self.datasets:
            d.set_batch_size(batch_size)

    def set_activation_probas(self, probas):
        """How frequent a dataset will produce dummy data to simulate
        unavailable input."""
        assert len(probas) != len(self.datasets)
        for ds, ap in zip(self.datasets, probas):
            ds.set_proba(ap)

    def rewind(self):
        for d in self.datasets:
            d.rewind()

    def next(self):
        for multi_data in zip(*self.datasets):
            # Get first OrderedDict
            f = multi_data[0]
            for d in multi_data[1:]:
                f.update(d)
            return f

        self.rewind()
        raise StopIteration

####################
# Some simple tests
####################

def test_combined_iterator():
    import cPickle
    from iter_text import TextIterator
    from iter_imgfeats import ImageFeatsIterator

    _dict = cPickle.load(open("/lium/trad4a/wmt/2016/caglayan/data/text/moses.tok/train.moses.tok.en.pkl"))
    imgiter = ImageFeatsIterator("/lium/trad4a/wmt/2016/caglayan/data/images/npy/fc7_vgg_feats_hdf5-flickr30k.valid.npy", batch_size=32)
    txtiter = TextIterator("/lium/trad4a/wmt/2016/caglayan/data/text/moses.tok/val.moses.tok.en",
                            _dict = _dict, n_words=0, do_mask=False, batch_size=32)
    txtiter.prepare_batches()

    miter = MultiIterator([imgiter, txtiter])
    next(miter)
    return miter

if __name__ == '__main__':
    miter = test_combined_iterator()
