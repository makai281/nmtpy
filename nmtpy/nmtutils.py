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

# Function to convert sentence to idxs
def sent_to_idx(vocab, tokens, limit=0):
    idxs = []
    for word in tokens:
        # Get token, 1 if not available
        idx = vocab.get(word, 1)
        if limit > 0:
            idx = idx if idx < limit else 1
        idxs.append(idx)
    return idxs

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
# Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
# "Exact solutions to the nonlinear dynamics of learning in deep
# linear neural networks." arXiv preprint arXiv:1312.6120 (2013).
def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(FLOAT)

# weight initializer, normal by default
def norm_weight(nin, nout, scale=0.01, ortho=True):
    if scale == "xavier":
        # Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks."
        # International conference on artificial intelligence and statistics. 2010.
        # http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf
        scale = 1. / np.sqrt(nin)
    elif scale == "he":
        # Claimed necessary for ReLU
        # Kaiming He et al. (2015)
        # Delving deep into rectifiers: Surpassing human-level performance on
        # imagenet classification. arXiv preprint arXiv:1502.01852.
        scale = 1. / np.sqrt(nin/2.)

    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(FLOAT)

################
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

    if n_samples == 1:
        x_mask = None

    return x, x_mask
