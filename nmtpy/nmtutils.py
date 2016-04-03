import numpy as np
import cPickle

from collections import OrderedDict
from .typedef import *

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_dictionary(fname):
    ivocab = {}
    with open(fname, "rb") as f:
        vocab = cPickle.load(f)

    for k,v in vocab.iteritems():
        ivocab[v] = k

    return vocab, ivocab

# Function to convert idxs to sentence
def idx_to_sent(ivocab, idxs):
    sent = []
    for widx in idxs:
        if widx == 0:
            break
        sent.append(ivocab.get(widx, "<unk>"))
    return " ".join(sent)


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)

# load parameters
def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Exception('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

# orthogonal initialization for weights
# see Saxe et al. ICLR'14
def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(FLOAT)

# weight initializer, normal by default
def norm_weight(nin, nout, scale=0.01, ortho=True):
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(FLOAT)

################
def mask_data(seqs, n_tsteps=-1):
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)

    # For ff-enc, we need fixed tsteps in the input
    maxlen = n_tsteps if n_tsteps > 0 else np.max(lengths) + 1

    # Shape is (t_steps, samples)
    x = np.zeros((maxlen, n_samples)).astype(INT)
    x_mask = np.zeros_like(x).astype(FLOAT)

    for idx, s_x in enumerate(seqs):
        x[:lengths[idx], idx] = s_x
        x_mask[:lengths[idx] + 1, idx] = 1.

    return x, x_mask
