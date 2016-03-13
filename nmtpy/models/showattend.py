#!/usr/bin/env python
from six.moves import range
from six.moves import zip

# Python
import os
import sys
import cPickle
import importlib

from collections import OrderedDict

# 3rd party
import numpy as np

import theano
import theano.tensor as tensor

# Ours
from nmtpy.layers import *
from nmtpy.typedef import *
from nmtpy.nmtutils import *
from nmtpy.search import gen_sample
from nmtpy.iterators import get_iterator
from nmtpy.models.basemodel import BaseModel

class Model(BaseModel):
    def __init__(self, trng, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(trng, **kwargs)

        # Load vocabularies if any
        if 'dicts' in kwargs:
            dicts = kwargs['dicts']
            assert 'trg' in dicts
            self.trg_dict, _ = load_dictionary(dicts['trg'])
            self.n_words_trg = min(self.n_words_trg, len(self.trg_dict)) if self.n_words_trg > 0 else len(self.trg_dict)

        # Convolutional feature dim
        self.n_convfeats = 512
        self.n_timesteps = 196 # 14x14 patches

        self.options = dict(self.__dict__)

    def load_data(self, shuffle=False, sort=False):
        ###############
        # Training data
        ###############

        # We need to find out about modalities
        data = self.options["data"]

        if 'train_idx' in data:
            # We have an idxs file for image ids
            image_idxs = open(data['train_idx']).read().strip().split("\n")
            self.image_idxs = [int(i) for i in image_idxs]

        train_src_type, train_src_file = data["train_src"]
        train_trg_type, train_trg_file = data["train_trg"]
        # src iter: img, trg iter: text
        train_src_iter = get_iterator(train_src_type)
        train_trg_iter = get_iterator(train_trg_type)

        # First load target texts
        train_trg_iterator = train_trg_iter(train_trg_file, self.trg_dict,
                                            batch_size=self.batch_size,
                                            n_words=self.n_words_trg,
                                            data_name='y', maxlen=self.maxlen,
                                            do_mask=True)
        train_trg_iterator.prepare_batches(shuffle=shuffle, sort=sort)
        batch_idxs = train_trg_iterator.get_idxs()

        # This should be image
        train_src_iterator = train_src_iter(train_src_file, batch_size=self.batch_size,
                                            idxs=batch_idxs, do_mask=False, n_timesteps=self.n_timesteps)
        train_src_iterator.prepare_batches()

        # Create multi iterator
        self.train_iterator = get_iterator("multi")([train_src_iterator, train_trg_iterator])

        #################
        # Validation data 
        #################

        if "valid_src" in data:
            # Validation data available
            valid_src_type, valid_src_file = data["valid_src"]
            valid_trg_type, valid_trg_file = data["valid_trg"]
            # src iter: img, trg iter: text
            valid_src_iter = get_iterator(valid_src_type)
            valid_trg_iter = get_iterator(valid_trg_type)

            # First load target texts
            valid_trg_iterator = valid_trg_iter(valid_trg_file, self.trg_dict,
                                                batch_size=self.batch_size,
                                                n_words=self.n_words_trg,
                                                data_name='y', maxlen=self.maxlen,
                                                do_mask=True)
            valid_trg_iterator.prepare_batches()
            batch_idxs = valid_trg_iterator.get_idxs()

            # This should be image
            valid_src_iterator = valid_src_iter(valid_src_file, batch_size=self.batch_size,
                                                idxs=batch_idxs, do_mask=False, n_timesteps=self.n_timesteps)
            valid_src_iterator.prepare_batches()

            # Create multi iterator
            self.valid_iterator = get_iterator("multi")([valid_src_iterator, valid_trg_iterator])

    def init_params(self):
        params = OrderedDict()

        # embedding weights for decoder
        params['Wemb_dec'] = norm_weight(self.n_words_trg, self.trg_emb_dim)

        # init_state, init_cell
        # n_convfeats (512) -> rnn_dim (1000)
        params = get_new_layer('ff')[0](params, prefix='ff_state', nin=self.n_convfeats, nout=self.rnn_dim)

        # decoder
        params = get_new_layer(self.dec_type)[0](params, prefix='decoder', nin=self.trg_emb_dim, dim=self.rnn_dim, dimctx=self.n_convfeats)

        # readout
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru'   , nin=self.rnn_dim, nout=self.trg_emb_dim, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_prev'  , nin=self.trg_emb_dim, nout=self.trg_emb_dim, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx'   , nin=self.n_convfeats, nout=self.trg_emb_dim, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit'       , nin=self.trg_emb_dim, nout=self.n_words_trg)

        self.initial_params = params

    def build(self):
        # Image features and all-1 mask
        # shape will be n_timesteps, n_samples, n_convfeats
        x_img = tensor.tensor3('x_img', dtype='float32')

        # Target sentences: n_timesteps, n_samples
        y = tensor.matrix('y', dtype=INT)
        y_mask = tensor.matrix('y_mask', dtype='float32')

        # Fixed # of timesteps for convolutional patches, e.g. 196
        n_timesteps = x_img.shape[0]

        # Volatile # of timesteps for target sentences
        n_timesteps_trg, n_samples = y.shape

        # Store tensors
        self.inputs['x_img'] = x_img
        self.inputs['y'] = y
        self.inputs['y_mask'] = y_mask

        # context is the convolutional vectors themselves
        # Take mean across the first axis which are the timesteps, e.g. conv patches
        # No need to multiply by all-one masks
        # ctx_mean: 1 x n_convfeat (512) x n_samples
        ctx_mean = tensor.mean(x_img, axis=0, dtype='float32')

        # initial decoder state: -> rnn_dim, e.g. 1000
        # NOTE: Try with linear activation as well
        # NOTE: we may need to normalize the features
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')

        # NOTE: Don't change afterwards here, the rest should be OK
        # word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        emb = self.tparams['Wemb_dec'][y.flatten()]
        emb = emb.reshape([n_timesteps_trg, n_samples, self.trg_emb_dim])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        # decoder - pass through the decoder conditional gru with attention
        # gru_cond returns hidden state, weighted sum of context vectors
        # and attentional weights.
        proj_h, ctxs, alphas = get_new_layer(self.dec_type)[1](self.tparams, emb,
                                                               prefix='decoder',
                                                               mask=y_mask, context=x_img,
                                                               context_mask=None,
                                                               one_step=False,
                                                               init_state=init_state)

        # rnn_dim -> trg_emb_dim
        logit_gru = get_new_layer('ff')[1](self.tparams, proj_h, prefix='ff_logit_gru', activ='linear')
        # trg_emb_dim -> trg_emb_dim
        logit_prev = get_new_layer('ff')[1](self.tparams, emb, prefix='ff_logit_prev', activ='linear')
        # ctx_dim -> trg_emb_dim
        logit_ctx = get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')

        # trg_emb_dim
        logit = tanh(logit_gru + logit_prev + logit_ctx)

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

        # trg_emb_dim -> n_words_trg
        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape

        # Apply logsoftmax (stable version)
        log_probs = -tensor.nnet.logsoftmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        # cost
        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * self.n_words_trg + y_flat

        # NOTE: We may want to normalize the cost by dividing
        # to the number of target tokens but this needs
        # scaling the learning rate accordingly.
        # cost = cost / y_mask.sum()
        cost = log_probs.flatten()[y_flat_idx]
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)

        # This computes the cost given the input tensors
        self.f_log_probs = theano.function(self.inputs.values(),
                                           cost,
                                           mode=self.func_mode,
                                           profile=self.profile)

        # Mean cost over batch
        self.cost = cost.mean()

        return self.cost

    def build_sampler(self):
        # shape will be n_timesteps, n_samples, n_convfeats
        x_img = tensor.tensor3('x_img', dtype='float32')
        n_timesteps = x_img.shape[0]
        n_samples = x_img.shape[1]

        # Take mean across the first axis which are the timesteps, e.g. conv patches
        # No need to multiply by all-one masks
        # ctx_mean: 1 x n_convfeat (512) x n_samples
        ctx_mean = tensor.mean(x_img, axis=0, dtype='float32')

        # initial decoder state: -> rnn_dim, e.g. 1000
        # NOTE: Try with linear activation as well
        # NOTE: we may need to normalize the features
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')

        # context is the convolutional vectors themselves
        ctx = x_img
        outs = [init_state, ctx]
        self.f_init = theano.function([x_img], outs, name='f_init', profile=self.profile)

        # x: 1 x 1
        y = tensor.vector('y_sampler', dtype=INT)
        init_state = tensor.matrix('init_state', dtype='float32')

        # if it's the first word, emb should be all zero and it is indicated by -1
        emb = tensor.switch(y[:, None] < 0,
                            tensor.alloc(0., 1, self.tparams['Wemb_dec'].shape[1]),
                            self.tparams['Wemb_dec'][y])

        # apply one step of conditional gru with attention
        proj = get_new_layer(self.dec_type)[1](self.tparams, emb,
                                                    prefix='decoder',
                                                    mask=None, context=ctx,
                                                    one_step=True,
                                                    init_state=init_state)
        # get the next hidden state
        next_state = proj[0]

        # get the weighted averages of context for this target word y
        ctxs = proj[1]

        logit_gru = get_new_layer('ff')[1](self.tparams, next_state, prefix='ff_logit_gru', activ='linear')
        logit_prev = get_new_layer('ff')[1](self.tparams, emb, prefix='ff_logit_prev', activ='linear')
        logit_ctx = get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')

        logit = tanh(logit_gru + logit_prev + logit_ctx)

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')

        # compute the softmax probability
        next_probs = tensor.nnet.softmax(logit)

        # sample from softmax distribution to get the sample
        next_sample = self.trng.multinomial(pvals=next_probs).argmax(1)

        # compile a function to do the whole thing above, next word probability,
        # sampled word for the next target, next hidden state to be used
        inputs = [y, ctx, init_state]
        outs = [next_probs, next_sample, next_state]
        self.f_next = theano.function(inputs, outs, name='f_next', profile=self.profile)
