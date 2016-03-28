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
        # shape will be n_timesteps, n_samples, n_convfeats (196, bs, 512)
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
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='linear')

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
        proj = get_new_layer(self.dec_type)[1](self.tparams, emb,
                                               prefix='decoder',
                                               mask=y_mask, context=x_img,
                                               context_mask=None,
                                               one_step=False,
                                               init_state=init_state)
        # gru_cond returns hidden state, weighted sum of context vectors
        # and attentional weights.
        proj_h = proj[0]
        ctxs = proj[1]
        alphas = proj[2]

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

        cost = log_probs.flatten()[y_flat_idx]
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)

        self.f_log_probs = theano.function(self.inputs.values(),
                                           cost,
                                           mode=self.func_mode,
                                           profile=self.profile)

        # We may want to normalize the cost by dividing
        # to the number of target tokens but this needs
        # scaling the learning rate accordingly.
        norm_cost = cost / y_mask.sum()

        return cost.mean(), norm_cost.mean()


    def build_sampler(self):
        # shape will be n_timesteps, n_samples, n_convfeats
        # context is the convolutional vectors themselves
        ctx = tensor.tensor3('x_img', dtype='float32')
        n_timesteps = ctx.shape[0]
        n_samples = ctx.shape[1]

        # Take mean across the first axis which are the timesteps, e.g. conv patches
        # No need to multiply by all-one masks
        # ctx_mean: 1 x n_convfeat (512) x n_samples
        ctx_mean = ctx.mean(0)

        # initial decoder state: -> rnn_dim, e.g. 1000
        # NOTE: Try with linear activation as well
        # NOTE: we may need to normalize the features
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='linear')

        # NOTE: No need to compute ctx as input is ctx as well.
        # But this will require changes in other parts of the code
        outs = [init_state, ctx]
        self.f_init = theano.function([ctx], outs, name='f_init', profile=self.profile)

        # x: 1 x 1
        y = tensor.vector('y_sampler', dtype=INT)
        init_state = tensor.matrix('init_state', dtype='float32')

        # if it's the first word, emb should be all zero and it is indicated by -1
        emb = tensor.switch(y[:, None] < 0,
                            tensor.alloc(0., 1, self.tparams['Wemb_dec'].shape[1]),
                            self.tparams['Wemb_dec'][y])

        # apply one step of conditional gru with attention
        r = get_new_layer(self.dec_type)[1](self.tparams, emb,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state)
        # get the next hidden state
        # get the weighted averages of context for this target word y
        next_state = r[0]
        ctxs = r[1]

        logit_gru = get_new_layer('ff')[1](self.tparams, next_state, prefix='ff_logit_gru', activ='linear')
        logit_prev = get_new_layer('ff')[1](self.tparams, emb, prefix='ff_logit_prev', activ='linear')
        logit_ctx = get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')

        logit = tanh(logit_gru + logit_prev + logit_ctx)

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')

        # compute the logsoftmax
        next_log_probs = tensor.nnet.logsoftmax(logit)

        # compile a function to do the whole thing above
        # sampled word for the next target, next hidden state to be used
        inputs = [y, ctx, init_state]
        outs = [next_log_probs, next_state]
        self.f_next = theano.function(inputs, outs, name='f_next', profile=self.profile)
