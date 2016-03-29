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
from ..sysutils import get_valid_evaluation, get_temp_file
from ..nmtutils import unzip, itemlist

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
        """Sets the seed for Theano RNG."""
        self.trng = RandomStreams(seed)

    def set_nanguard(self):
        """Enables nanguard for theano functions."""
        self.func_mode = None
        if self.nanguard:
            from theano.compile.nanguardmode import NanGuardMode
            self.func_mode = NanGuardMode(nan_is_error=True,
                                          inf_is_error=True,
                                          big_is_error=False)

    def set_dropout(self, val):
        """Sets dropout indicator for activation scaling
        if dropout is available through configuration."""
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

    def add_l2_weight_decay(self, cost, decay_c):
        decay_c = theano.shared(np.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for _, vv in self.tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
        return cost

    def add_alpha_regularizer(self, cost, alpha_c):
        # This should be reimplemented in attentional models
        return cost

    def build_optimizer(self, cost, clip_c):
        grads = tensor.grad(cost, wrt=itemlist(self.tparams))
        if clip_c > 0.:
            g2 = 0.
            new_grads = []
            for g in grads:
                g2 += (g**2).sum()
            for g in grads:
                new_grads.append(tensor.switch(g2 > (clip_c**2),
                                               g / tensor.sqrt(g2) * clip_c,
                                               g))
            grads = new_grads

        opt = importlib.import_module("nmtpy.optimizers").__dict__[self.optimizer]
        lr = tensor.scalar(name='lr')
        self.f_grad_shared, self.f_update = opt(lr, self.tparams,
                                                grads, self.inputs.values(),
                                                cost, profile=self.profile,
                                                mode=self.func_mode)

    def run_beam_search(self, beam_size=12, n_jobs=8, metric='bleu', out_file=None):
        # Save model temporarily
        with get_temp_file(suffix=".npz", delete=True) as tmpf:
            self.save_params(tmpf.name, **unzip(self.tparams))

            result = get_valid_evaluation(tmpf.name,
                                          pkl_path=self.model_path + ".pkl",
                                          beam_size=beam_size,
                                          n_jobs=n_jobs,
                                          metric=metric,
                                          out_file=out_file)

        return result

    def generate_samples(self, batch_dict, n):
        # If you implement this in your class and set sample_freq
        # accordingly in the model configuration, the training loop
        # will periodically sample translations for debugging
        # or informational purposes. If you don't, this is a NOP.
        pass

    ##########################################################
    # For all the abstract methods below, you can take a look
    # at attention.py to understand how they are implemented.
    # Remember that you have to implement these methods in your
    # own model.
    ##########################################################

    @abstractmethod
    def load_data(self):
        # Load and prepare your training and validation data
        # inside this function.
        pass

    @abstractmethod
    def init_params(self):
        # Initialize the weights and biases of your network
        # through the helper functions provided in layers.py.
        pass

    @abstractmethod
    def build(self):
        # This builds the computational graph of your network.
        pass

    @abstractmethod
    def build_sampler(self):
        # This is quite similar to build() but works in a
        # sequential manner for beam-search or sampling.
        pass

    @abstractmethod
    def beam_search(self, inputs):
        # Beam search can change a lot based on the RNN
        # layer, types of input etc. Look at the attention model
        # and copy it into your class and modify it correctly.

        # nmt-translate will also used the relevant beam_search
        # based on the model type.
        pass
