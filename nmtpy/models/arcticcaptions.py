#!/usr/bin/env python
from six.moves import range
from six.moves import zip

# Python
import os
import cPickle
import importlib

from collections import OrderedDict

# 3rd party
import numpy as np

import theano
import theano.tensor as tensor

# Ours
from ..layers import *
from ..typedef import *
from ..nmtutils import *
from ..iterators import get_iterator

from ..models.basemodel import BaseModel

class Model(BaseModel):
    def __init__(self, seed, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(**kwargs)

        # Load vocabularies if any
        if 'dicts' in kwargs:
            dicts = kwargs['dicts']
            assert 'trg' in dicts
            self.trg_dict, trg_idict = load_dictionary(dicts['trg'])
            self.n_words_trg = min(self.n_words_trg, len(self.trg_dict)) if self.n_words_trg > 0 else len(self.trg_dict)

        # Convolutional feature dim
        self.n_convfeats = 512
        self.n_timesteps = 196 # 14x14 patches

        # Collect options
        self.options = dict(self.__dict__)
        self.trg_idict = trg_idict

        self.set_nanguard()
        self.set_trng(seed)

        # Set iterator types here
        self.train_src_iter = "img_feats"
        self.train_trg_iter = "text"
        self.valid_src_iter = "img_feats"
        self.valid_trg_iter = "text"

    def load_data(self):
        # Do we have an idxs file for image ids?
        image_idxs = None
        if 'train_idx' in self.data:
            image_idxs = open(self.data['train_idx']).read().strip().split("\n")
            image_idxs = [int(i) for i in image_idxs]

        # First load target texts
        train_trg_iterator = get_iterator(self.train_trg_iter)(
                                    self.data['train_trg'], self.trg_dict,
                                    batch_size=self.batch_size,
                                    n_words=self.n_words_trg,
                                    data_name='y',
                                    do_mask=True)
        train_trg_iterator.prepare_batches()

        # This should be image
        train_src_iterator = get_iterator(self.train_src_iter)(
                                    self.data['train_src'],
                                    batch_size=self.batch_size,
                                    idxs=image_idxs,
                                    do_mask=False,
                                    n_timesteps=self.n_timesteps)

        # Create multi iterator
        self.train_iterator = get_iterator("multi")([train_src_iterator, train_trg_iterator])

        #################
        # Validation data 
        #################
        valid_batch_size = 64

        valid_trg_files = self.data['valid_trg']
        if isinstance(valid_trg_files, str):
            valid_trg_files = list([valid_trg_files])

        # First load target texts
        valid_trg_iterator = get_iterator(self.valid_trg_iter)(
                                    valid_trg_files[0], self.trg_dict,
                                    batch_size=valid_batch_size,
                                    n_words=self.n_words_trg,
                                    data_name='y',
                                    do_mask=True)
        valid_trg_iterator.prepare_batches()

        # This is image features
        valid_src_iterator = get_iterator(self.valid_src_iter)(
                                    self.data['valid_src'],
                                    batch_size=valid_batch_size,
                                    idxs=None,
                                    do_mask=False,
                                    n_timesteps=self.n_timesteps)

        # Create multi iterator
        self.valid_iterator = get_iterator("multi")([valid_src_iterator, valid_trg_iterator])

    def init_params(self):
        params = OrderedDict()

        # embedding: [matrix E in paper]
        params['Wemb'] = norm_weight(self.n_words_trg, self.trg_emb_dim)
        ctx_dim = self.ctx_dim

        # init_state, init_cell: [top right on page 4]
        for lidx in xrange(1, self.n_layers_init):
            params = get_layer('ff')[0](options, params, prefix='ff_init_%d'%lidx, nin=ctx_dim, nout=ctx_dim)
        params = get_layer('ff')[0](options, params, prefix='ff_state', nin=ctx_dim, nout=options['dim'])
        params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=ctx_dim, nout=options['dim'])
        # decoder: LSTM: [equation (1)/(2)/(3)]
        params = get_layer('lstm_cond')[0](options, params, prefix='decoder',
                                           nin=options['dim_word'], dim=options['dim'],
                                           dimctx=ctx_dim)
        # potentially deep decoder (warning: should work but somewhat untested)
        if options['n_layers_lstm'] > 1:
            for lidx in xrange(1, options['n_layers_lstm']):
                params = get_layer('ff')[0](options, params, prefix='ff_state_%d'%lidx, nin=options['ctx_dim'], nout=options['dim'])
                params = get_layer('ff')[0](options, params, prefix='ff_memory_%d'%lidx, nin=options['ctx_dim'], nout=options['dim'])
                params = get_layer('lstm_cond')[0](options, params, prefix='decoder_%d'%lidx,
                                                   nin=options['dim'], dim=options['dim'],
                                                   dimctx=ctx_dim)
        # readout: [equation (7)]
        params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], nout=options['dim_word'])
        if options['ctx2out']:
            params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx', nin=ctx_dim, nout=options['dim_word'])
        if options['n_layers_out'] > 1:
            for lidx in xrange(1, options['n_layers_out']):
                params = get_layer('ff')[0](options, params, prefix='ff_logit_h%d'%lidx, nin=options['dim_word'], nout=options['dim_word'])
        params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim_word'], nout=options['n_words'])



        self.initial_params = params

    def build(self):
        # description string: #words x #samples,
        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype=FLOAT)
        # context: #samples x #annotations x dim
        ctx = tensor.tensor3('ctx', dtype=FLOAT)

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # index into the word embedding matrix, shift it forward in time
        emb = self.tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, self.trg_emb_dim)
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted
        # NOTE: Missing lstm_encoder part
        ctx0 = ctx

        # initial state/cell [top right on page 4]
        ctx_mean = ctx0.mean(1)
        for lidx in xrange(1, self.n_layers_init):
            ctx_mean = get_layer('ff')[1](self.tparams, ctx_mean,
                                          prefix='ff_init_%d' % lidx, activ='rectifier')
            if self.dropout > 0:
                ctx_mean = dropout_layer(ctx_mean, self.use_dropout, self.dropout, self.trng)

        init_state = get_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')
        init_memory = get_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_memory', activ='tanh')
        # lstm decoder
        # [equation (1), (2), (3) in section 3.1.2]
        attn_updates = []
        proj, updates = get_layer('lstm_cond')[1](self.tparams, emb, options,
                                                  prefix='decoder',
                                                  mask=mask, context=ctx0,
                                                  one_step=False,
                                                  init_state=init_state,
                                                  init_memory=init_memory,
                                                  trng=self.trng,
                                                  use_noise=self.use_dropout, # FIXME
                                                  sampling=sampling)
        attn_updates += updates
        proj_h = proj[0]

        # optional deep attention
        if self.n_layers_lstm > 1:
            for lidx in xrange(1, self.n_layers_lstm):
                init_state = get_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state_%d'%lidx, activ='tanh')
                init_memory = get_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_memory_%d'%lidx, activ='tanh')
                proj, updates = get_layer('lstm_cond')[1](tparams, proj_h, options,
                                                          prefix='decoder_%d' % lidx,
                                                          mask=mask, context=ctx0,
                                                          one_step=False,
                                                          init_state=init_state,
                                                          init_memory=init_memory,
                                                          trng=trng,
                                                          use_noise=self.use_dropout,
                                                          sampling=sampling)
                attn_updates += updates
                proj_h = proj[0]

        alphas       = proj[2]
        alpha_sample = proj[3]
        ctxs         = proj[4]

        # [beta value explained in note 4.2.1 "doubly stochastic attention"]
        if self.selector:
            sels = proj[5]

        if self.dropout > 0:
            proj_h = dropout_layer(proj_h, self.use_dropout, self.dropout, self.trng)

        # compute word probabilities
        # [equation (7)]
        logit = get_layer('ff')[1](self.tparams, proj_h, prefix='ff_logit_lstm', activ='linear')
        if self.prev2out:
            logit += emb

        if self.ctx2out:
            logit += get_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')

        logit = tanh(logit)

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

        if self.n_layers_out > 1:
            for lidx in xrange(1, self.n_layers.out):
                logit = get_layer('ff')[1](self.tparams, logit, prefix='ff_logit_h%d' % lidx, activ='rectifier')
            if self.dropout > 0:
                logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

        # compute softmax
        logit = get_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape
        log_probs = -tensor.nnet.logsoftmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        # Index into the computed probability to give the log likelihood
        x_flat = x.flatten()
        log_p_flat = probs.flatten()
        cost = log_p_flat[tensor.arange(x_flat.shape[0])*probs.shape[1]+x_flat]
        cost = cost.reshape([x.shape[0], x.shape[1]])
        masked_cost = cost * mask
        cost = (masked_cost).sum(0)

        # optional outputs
        opt_outs = dict() 
        if options['selector']:
            opt_outs['selector'] = sels

        return trng, use_noise, [x, mask, ctx], alphas, alpha_sample, cost, opt_outs


        # Store tensors
        self.inputs['x_img'] = x_img
        self.inputs['y'] = y
        self.inputs['y_mask'] = y_mask

        return cost.mean()
