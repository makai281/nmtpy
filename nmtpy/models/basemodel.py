from six.moves import range
from six.moves import zip

import os
import cPickle
import inspect
import importlib

from collections import OrderedDict
from hashlib import sha1

from abc import ABCMeta, abstractmethod

import theano
import theano.tensor as tensor

import numpy as np

class BaseModel(object):
    __metaclass__ = ABCMeta
    def __init__(self, trng, **kwargs):
        # This is for tracking changes in the source code
        #self.__base_version = sha1(inspect.getsource(self.__class__)).hexdigest() 
        self.__dict__.update(kwargs)

        self.trng = trng
        self.name = os.path.splitext(os.path.basename(self.model_path))[0]

        self.do_dropout = True if self.dropout > 0 else False
        self.use_dropout = theano.shared(np.float32(0.))

        # Input tensor lists
        self.inputs = OrderedDict()

        # Theano variables
        self.f_log_probs = None
        self.cost = None
        self.f_init = None
        self.f_next = None
        self.f_update = None
        self.f_grad_shared = None

        self.params = None
        self.tparams = None

        # Iterators
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        self.func_mode = None
        if self.nanguard:
            from theano.compile.nanguardmode import NanGuardMode
            self.func_mode = NanGuardMode(nan_is_error=True,
                                          inf_is_error=True,
                                          big_is_error=True)

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
        for kk, pp in self.params.iteritems():
            self.tparams[kk] = theano.shared(self.params[kk], name=kk)

    def val_loss(self):
        probs = []

        # dict of x, x_mask, y, y_mask
        for data in self.valid_iterator:
            probs.extend(self.f_log_probs(*data.values()))

        return np.array(probs).mean()

    def build_optimizer(self, grads):
        opt = importlib.import_module("nmtpy.optimizers").__dict__[self.optimizer]
        lr = tensor.scalar(name='lr')
        self.f_grad_shared, self.f_update = opt(lr, self.tparams,
                                                grads, self.inputs.values(),
                                                self.cost, profile=self.profile,
                                                mode=self.func_mode)
        #self.f_grad_shared.trust_input = True
        #self.f_update.trust_input = True

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
