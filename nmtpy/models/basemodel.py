from six.moves import range
from six.moves import zip

import os
import inspect
import importlib

from collections import OrderedDict

from abc import ABCMeta, abstractmethod

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
from ..nmtutils import unzip, get_param_dict
from ..sysutils import *
from ..typedef import *

#######################################
## For debugging function input outputs
def inspect_inputs(i, node, fn):
    print '>> Inputs: ', i, node, [input[0] for input in fn.inputs]

def inspect_outputs(i, node, fn):
    print '>> Outputs: ', i, node, [input[0] for input in fn.outputs]
#######################################

class BaseModel(object):
    __metaclass__ = ABCMeta
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # Will be set when set_dropout is first called
        self.use_dropout = None

        # Input tensor lists
        self.inputs = OrderedDict()

        # Theano variables
        self.f_log_probs = None
        self.f_init = None
        self.f_next = None

        self.initial_params = None
        self.tparams = None

        # Iterators
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        # Will be a theano shared variable for lrate annealing
        self.learning_rate = None

    def set_options(self, optdict):
        """Filter out None's and save option dict."""
        self.options = OrderedDict([(k,v) for k,v in optdict.items() if v is not None])

    def set_trng(self, seed):
        """Sets the seed for Theano RNG."""
        self.trng = RandomStreams(seed)

    def set_dropout(self, val):
        """Sets dropout indicator for activation scaling
        if dropout is available through configuration."""
        if self.use_dropout is None:
            self.use_dropout = theano.shared(np.float32(0.))
        else:
            self.use_dropout.set_value(float(val))

    def update_lrate(self, lrate):
        """Update learning rate."""
        self.learning_rate.set_value(lrate)

    def get_nb_params(self):
        """Returns the number of parameters of the model."""
        total = 0
        for p in self.initial_params.values():
            total += p.size
        return readable_size(total)

    def set_shared_variables(self, updates):
        for k in self.tparams.keys():
            self.tparams[k].set_value(updates[k])

    def save(self, fname):
        if self.tparams is not None:
            np.savez(fname, tparams=unzip(self.tparams), opts=self.options)
        else:
            np.savez(fname, opts=self.options)

    def load(self, fname):
        self.tparams = OrderedDict()

        params = get_param_dict(fname)
        for k,v in params.iteritems():
            self.tparams[k] = theano.shared(v, name=k)

    def init_shared_variables(self, _from=None):
        # initialize Theano shared variables according to the _from
        # _from can be given as a pre-trained model from nmt-train
        if _from is None:
            _from = self.initial_params

        # tparams is None for the first call
        if self.tparams is None:
            self.tparams = OrderedDict()
            for kk, pp in _from.iteritems():
                self.tparams[kk] = theano.shared(_from[kk], name=kk)
        else:
            # Already initialized the params, override them
            for kk in self.tparams.keys():
                # Let this fail if _from doesn't match the model
                self.tparams[kk].set_value(_from[kk])

    def val_loss(self, sentence=None):
        probs = []
        if sentence is None:
            for data in self.valid_iterator:
            # Don't fail if data doesn't contain y_mask. The loss won't
            # be normalized but the training will continue
            #NOTE: y_mask is now mandatory even if batch size is 1
                norm = data['y_mask'].sum(0)
                log_probs = sum(self.f_log_probs(*data.values())) / norm
                probs.extend(log_probs)
            return np.array(probs).mean()
        else:
            norm = sentence['y_mask'].sum(0)
            log_probs = self.f_log_probs_detailled(*sentence.values())
            probs.extend(log_probs)
            return np.array(probs), norm

    def get_l2_weight_decay(self, decay_c, skip_bias=True):
        decay_c = theano.shared(np.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in self.tparams.iteritems():
            # Skip biases for L2 regularization
            if not skip_bias or (skip_bias and vv.get_value().ndim > 1):
                weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        return weight_decay

    def get_regularized_cost(self, cost, decay_c, alpha_c=None):
        reg_cost = cost
        if decay_c > 0:
            reg_cost += self.get_l2_weight_decay(decay_c)
        if alpha_c and alpha_c > 0:
            reg_cost += self.get_alpha_regularizer(cost, alpha_c)
        return reg_cost

    def get_clipped_grads(self, grads, clip_c):
        # Gradient clipping
        g2 = 0.
        new_grads = []
        for g in grads:
            g2 += (g**2).sum()
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        return new_grads

    def build_optimizer(self, cost, clip_c, dont_update=None, debug=False):
        tparams = OrderedDict(self.tparams)

        if dont_update is not None:
            for key in tparams:
                if key in dont_update:
                    del tparams[key]

        # Get gradients of cost with respect to variables
        grads = tensor.grad(cost, wrt=tparams.values())

        if clip_c > 0:
            grads = self.get_clipped_grads(grads, clip_c)

        #self.grad_norm = theano.function(self.inputs.values(),
        #                                 tensor.sqrt(sum([(g**2).sum() for g in grads])))

        # Load optimizer
        opt = importlib.import_module("nmtpy.optimizers").__dict__[self.optimizer]

        # Create theano shared variable for learning rate
        self.learning_rate = theano.shared(np.float32(self.lrate), name='lrate')

        # Get updates
        updates = opt(tparams, grads, self.inputs.values(),
                      cost, lr0=self.learning_rate)

        # Compile forward/backward function
        if debug:
            self.train_batch = theano.function(self.inputs.values(), cost, updates=updates,
                                               mode=theano.compile.MonitorMode(
                                                   pre_func=inspect_inputs,
                                                   post_func=inspect_outputs))
        else:
            self.train_batch = theano.function(self.inputs.values(), cost, updates=updates)

    def run_beam_search(self, beam_size=12, n_jobs=8, metric='bleu', mode='beamsearch', out_file=None):
        # Save model temporarily
        with get_temp_file(suffix=".npz", delete=True) as tmpf:
            self.save(tmpf.name)
            result = get_valid_evaluation(tmpf.name,
                                          beam_size=beam_size,
                                          n_jobs=n_jobs,
                                          metric=metric,
                                          mode=mode,
                                          out_file=out_file)

        return result

    def gen_sample(self, input_dict, maxlen=50, argmax=False):
        # A method that samples or takes the max proba's or
        # does a forced decoding depending on the parameters.
        final_sample = []
        final_score = 0

        target = None
        if "y_true" in input_dict:
            # We're doing forced decoding
            target = input_dict.pop("y_true")
            maxlen = len(target)

        inputs = input_dict.values()

        next_state, ctx0 = self.f_init(*inputs)

        # Beginning-of-sentence indicator is -1
        next_word = np.array([-1], dtype=INT)

        for ii in xrange(maxlen):
            # Get next states
            next_log_p, next_word, next_state = self.f_next(*[next_word, ctx0, next_state])

            if target is not None:
                nw = int(target[ii])

            elif argmax:
                # argmax() works the same for both probas and log_probas
                nw = next_log_p[0].argmax()

            else:
                # Multinomial sampling
                nw = next_word[0]

            if nw == 0:
                break

            # Add the word idx
            final_sample.append(nw)
            final_score -= next_log_p[0, nw]

        final_sample = [final_sample]
        final_score = np.array(final_score)

        return final_sample, final_score

    def generate_samples(self, batch_dict, n_samples):
        # Silently fail if generate_samples is not reimplemented
        # in child classes
        return None

    def info(self):
        pass

    def get_alpha_regularizer(self, alpha_c):
        # This should be implemented in attentional models if necessary.
        return 0.

    def beam_search(self, inputs, beam_size=12, maxlen=50, suppress_unks=False, **kwargs):
        # Beam search can change a lot based on the RNN
        # layer, types of input etc. Look at the attention model
        # and copy it into your class and modify it correctly.

        # nmt-translate will also used the relevant beam_search
        # based on the model type.

        # You can pass additional arguments to beam_search through kwargs.
        pass

    ##########################################################
    # For all the abstract methods below, you can take a look
    # at attention.py to understand how they are implemented.
    # Remember that you NEED to implement these methods in your
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
