from six.moves import range
from six.moves import zip

import os
import cPickle
import inspect
import importlib

from collections import OrderedDict

from abc import ABCMeta, abstractmethod

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
from ..sysutils import get_valid_evaluation
from ..nmtutils import unzip

class BaseModel(object):
    __metaclass__ = ABCMeta
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.name = os.path.splitext(os.path.basename(self.model_path))[0]

        self.do_dropout = True if self.dropout > 0 else False
        self.use_dropout = theano.shared(np.float32(0.))

        # Input tensor lists
        self.inputs = OrderedDict()

        # Theano variables
        self.f_log_probs = None
        self.f_init = None
        self.f_next = None
        self.f_update = None
        self.f_grad_shared = None

        self.initial_params = None
        self.tparams = None

        # Iterators
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

    def set_trng(self, seed):
        self.trng = RandomStreams(seed)

    def set_nanguard(self):
        self.func_mode = None
        if self.nanguard:
            from theano.compile.nanguardmode import NanGuardMode
            self.func_mode = NanGuardMode(nan_is_error=True,
                                          inf_is_error=True,
                                          big_is_error=False)

    def set_dropout(self, val):
        # Enable use_dropout in training. (Effective if dropout exists)
        self.use_dropout.set_value(float(val))

    def load_params(self, params):
        self.tparams = OrderedDict()
        for k,v in params.iteritems():
            # FIXME: Hack to avoid these params to appear
            if not k.startswith(("uidx", "zipped", "valid_history", "bleu_history")):
                self.tparams[k] = theano.shared(v, name=k)

    def save_params(self, fname, **kwargs):
        np.savez(fname, **kwargs)

    def save_options(self, filepath=None):
        if not filepath:
            filepath = self.model_path + ".pkl"

        with open(filepath, 'wb') as f:
            cPickle.dump(self.options, f, cPickle.HIGHEST_PROTOCOL)

    def init_shared_variables(self):
        # initialize Theano shared variables according to the initial parameters
        self.tparams = OrderedDict()
        for kk, pp in self.initial_params.iteritems():
            self.tparams[kk] = theano.shared(self.initial_params[kk], name=kk)

    def val_loss(self):
        probs = []

        # dict of x, x_mask, y, y_mask
        for data in self.valid_iterator:
            probs.extend(self.f_log_probs(*data.values()))

        return np.array(probs).mean()

    def build_optimizer(self, cost, grads):
        opt = importlib.import_module("nmtpy.optimizers").__dict__[self.optimizer]
        lr = tensor.scalar(name='lr')
        self.f_grad_shared, self.f_update = opt(lr, self.tparams,
                                                grads, self.inputs.values(),
                                                cost, profile=self.profile,
                                                mode=self.func_mode)

    def beam_search(self, beam_size=12):
        tmp_model = os.path.join("/tmp", self.name) + ".npz"
        tmp_opts = "%s.pkl" % tmp_model
        # Save model temporarily
        self.save_params(tmp_model, **unzip(self.tparams))
        self.save_options(filepath=tmp_opts)
        result = get_valid_evaluation(tmp_model, beam_size)
        os.unlink(tmp_model)
        os.unlink(tmp_opts)
        return result

    @abstractmethod
    def load_data(self, shuffle=False, sort=False):
        pass

    @abstractmethod
    def init_params(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def build_sampler(self):
        pass

    def generate_samples(self, batch_dict, n):
        pass
