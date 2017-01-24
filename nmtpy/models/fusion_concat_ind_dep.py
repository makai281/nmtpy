# -*- coding: utf-8 -*-
from six.moves import range
from six.moves import zip

# Python
import os
import copy
import cPickle
import importlib

from collections import OrderedDict

# 3rd party
import numpy as np

import theano
import theano.tensor as tensor

# Ours
from ..layers import *
from ..defaults import INT, FLOAT
from ..nmtutils import *
from ..iterators.wmt import WMTIterator

from .attention import Model as ParentModel

########################################################
# IND-DEP Attention (att shared, dec distinct) Mechanism
########################################################
def init_gru_decoder_multi(params, nin, dim, dimctx, scale=0.01, prefix='gru_decoder_multi'):
    # Init with usual gru_cond function
    params = param_init_gru_cond(params, nin, dim, dimctx, scale, prefix)

    # Add weights for concat fusion
    params[pp(prefix, 'W_fus')] = norm_weight(2*dimctx, dimctx, scale=scale)
    params[pp(prefix, 'c_fus')] = np.zeros((dimctx, )).astype(FLOAT)

    # Add separate attention weights for the 2nd modality
    params[pp(prefix, 'Wc_att2')] = norm_weight(dimctx, dimctx, scale=scale)
    params[pp(prefix,  'b_att2')] = np.zeros((dimctx,)).astype(FLOAT)

    params[pp(prefix, 'W_comb_att2')] = norm_weight(dim, dimctx, scale=scale)
    return params

def gru_decoder_multi(tparams, state_below,
                      ctx1, ctx2, prefix='gru_decoder_multi',
                      input_mask=None, one_step=False,
                      init_state=None, ctx1_mask=None):
    if one_step:
        assert init_state, 'previous state must be provided'

    # Context
    # n_timesteps x n_samples x ctxdim
    assert ctx1 and ctx2, 'Contexts must be provided'
    assert ctx1.ndim == 3 and ctx2.ndim == 3, 'Contexts must be 3-d: #annotation x #sample x dim'

    # Number of padded source timesteps
    nsteps = state_below.shape[0]

    # Batch or single sample?
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1

    # if we have no mask, we assume all the inputs are valid
    # tensor.alloc(value, *shape)
    # input_mask: (n_steps, 1) filled with 1
    if input_mask is None:
        input_mask = tensor.alloc(1., nsteps, 1)

    # Infer RNN dimensionality
    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    # initial/previous state
    # if not given, assume it's all zeros
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # These two dot products are same with gru_layer, refer to the equations.
    # [W_r * X + b_r, W_z * X + b_z]
    state_below_ = tensor.dot(state_below, tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')]

    # input to compute the hidden state proposal
    # This is the [W*x]_j in the eq. 8 of the paper
    state_belowx = tensor.dot(state_below, tparams[pp(prefix, 'Wx')]) + tparams[pp(prefix, 'bx')]

    # Wc_att: dimctx -> dimctx
    # Linearly transform the contexts to another space with same dimensionality
    pctx1_ = tensor.dot(ctx1, tparams[pp(prefix, 'Wc_att')]) + tparams[pp(prefix, 'b_att')]
    pctx2_ = tensor.dot(ctx2, tparams[pp(prefix, 'Wc_att2')]) + tparams[pp(prefix, 'b_att2')]

    # Step function for the recurrence/scan
    # Sequences
    # ---------
    # m_    : mask
    # x_    : state_below_
    # xx_   : state_belowx
    # outputs_info
    # ------------
    # h_     : init_state,
    # ctx_   : need to be defined as it's returned by _step
    # alpha1_: need to be defined as it's returned by _step
    # alpha2_: need to be defined as it's returned by _step
    # non sequences
    # -------------
    # pctx1_ : pctx1_
    # pctx2_ : pctx2_
    # cc1_   : ctx1
    # cc2_   : ctx2
    # and all the shared weights and biases..
    def _step(m_, x_, xx_,
              h_, ctx_, alpha1_, alpha2_, # These ctx and alpha's are not used in the computations
              pctx1_, pctx2_, cc1_, cc2_, U, Wc, W_comb_att, W_comb_att2, U_att, c_att,
              Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl, W_fus, c_fus):

        # Do a step of classical GRU
        h1 = gru_step(m_, x_, xx_, h_, U, Ux)

        ####################################################
        # NOTE: Shared Attention with distinct decoder state
        ####################################################
        # h1 X W_comb_att
        # W_comb_att: dim -> dimctx
        # pstate_ should be 2D as we're working with unrolled timesteps
        pstate1_ = tensor.dot(h1, W_comb_att)
        pstate2_ = tensor.dot(h1, W_comb_att2)

        # Accumulate in pctx*__ and apply tanh()
        # This becomes the projected context(s) + the current hidden state
        # of the decoder, e.g. this is the information accumulating
        # into the returned original contexts with the knowledge of target
        # sentence decoding.
        pctx1__ = tanh(pctx1_ + pstate1_[None, :, :])
        pctx2__ = tanh(pctx2_ + pstate2_[None, :, :])

        # Affine transformation for alpha* = (pctx*__ X U_att) + c_att
        # We're now down to scalar alpha's for each accumulated
        # context (0th dim) in the pctx*__
        # alpha1 should be n_timesteps, 1, 1
        alpha1 = tensor.dot(pctx1__, U_att) + c_att
        alpha2 = tensor.dot(pctx2__, U_att) + c_att

        # Drop the last dimension, e.g. (n_timesteps, 1)
        alpha1 = alpha1.reshape([alpha1.shape[0], alpha1.shape[1]])
        alpha2 = alpha2.reshape([alpha2.shape[0], alpha2.shape[1]])

        # Exponentiate alpha1
        alpha1 = tensor.exp(alpha1 - alpha1.max(0, keepdims=True))
        alpha2 = tensor.exp(alpha2 - alpha2.max(0, keepdims=True))

        # If there is a context mask, multiply with it to cancel unnecessary steps
        # We won't have a ctx_mask for image vectors
        if ctx1_mask:
            alpha1 = alpha1 * ctx1_mask

        # Normalize so that the sum makes 1
        alpha1 = alpha1 / alpha1.sum(0, keepdims=True)
        alpha2 = alpha2 / alpha2.sum(0, keepdims=True)

        # Compute the current context ctx*_ as the alpha-weighted sum of
        # the initial contexts ctx*'s
        ctx1_ = (cc1_ * alpha1[:, :, None]).sum(0)
        ctx2_ = (cc2_ * alpha2[:, :, None]).sum(0)
        # n_samples x ctxdim (2000)

        ##############################################
        # NOTE: This is the fusion context with concat
        ##############################################
        ctx_ = tensor.dot(tensor.concatenate([ctx1_, ctx2_], axis=1), W_fus) + c_fus

        ############################################
        # ctx*_ and alpha computations are completed
        ############################################

        ####################################
        # The below code is another GRU cell
        ####################################
        # Affine transformation: h1 X U_nl + b_nl
        # U_nl, b_nl: Stacked dim*2
        preact = tensor.dot(h1, U_nl) + b_nl

        # Transform the weighted context sum with Wc
        # and add it to preact
        # Wc: dimctx -> Stacked dim*2
        preact += tensor.dot(ctx_, Wc)

        # Apply sigmoid nonlinearity
        preact = sigmoid(preact)

        # Slice activations: New gates r2 and u2
        r2 = tensor_slice(preact, 0, dim)
        u2 = tensor_slice(preact, 1, dim)

        preactx = (tensor.dot(h1, Ux_nl) + bx_nl) * r2
        preactx += tensor.dot(ctx_, Wcx)

        # Candidate hidden
        h2_tilda = tanh(preactx)

        # Leaky integration between the new h2 and the
        # old h1 computed in line 285
        h2 = u2 * h2_tilda + (1. - u2) * h1
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha1.T, alpha2.T

    # Sequences are the input mask and the transformed target embeddings
    seqs = [input_mask, state_below_, state_belowx]

    # Create a list of shared parameters for easy parameter passing
    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Wc')],
                   tparams[pp(prefix, 'W_comb_att')],
                   tparams[pp(prefix, 'W_comb_att2')],
                   tparams[pp(prefix, 'U_att')],
                   tparams[pp(prefix, 'c_att')],
                   tparams[pp(prefix, 'Ux')],
                   tparams[pp(prefix, 'Wcx')],
                   tparams[pp(prefix, 'U_nl')],
                   tparams[pp(prefix, 'Ux_nl')],
                   tparams[pp(prefix, 'b_nl')],
                   tparams[pp(prefix, 'bx_nl')],
                   tparams[pp(prefix, 'W_fus')],
                   tparams[pp(prefix, 'c_fus')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, None, pctx1_, pctx2_, ctx1, ctx2] + shared_vars))
    else:
        outputs_info=[init_state,
                      tensor.alloc(0., n_samples, ctx1.shape[2]), # ctxdim       (ctx_)
                      tensor.alloc(0., n_samples, ctx1.shape[0]), # n_timesteps  (alpha1)
                      tensor.alloc(0., n_samples, ctx2.shape[0])] # n_timesteps  (alpha2)

        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=[pctx1_, pctx2_, ctx1, ctx2] + shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    strict=True)
    return rval

class Model(ParentModel):
    def __init__(self, seed, logger, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(seed, logger, **kwargs)

    def info(self):
        self.logger.info('Source vocabulary size: %d', self.n_words_src)
        self.logger.info('Target vocabulary size: %d', self.n_words_trg)
        self.logger.info('%d training samples' % self.train_iterator.n_samples)
        self.logger.info('  %d/%d UNKs in source, %d/%d UNKs in target' % (self.train_iterator.unk_src,
                                                                          self.train_iterator.total_src_words,
                                                                          self.train_iterator.unk_trg,
                                                                          self.train_iterator.total_trg_words))
        self.logger.info('%d validation samples' % self.valid_iterator.n_samples)
        self.logger.info('  %d UNKs in source' % self.valid_iterator.unk_src)

    def load_data(self):
        # Load training data
        self.train_iterator = WMTIterator(
                batch_size=self.batch_size,
                shuffle_mode=self.smode,
                logger=self.logger,
                pklfile=self.data['train_src'],
                imgfile=self.data['train_img'],
                trgdict=self.trg_dict,
                srcdict=self.src_dict,
                n_words_trg=self.n_words_trg, n_words_src=self.n_words_src,
                mode=self.options.get('data_mode', 'pairs'))
        self.train_iterator.read()
        self.load_valid_data()

    def load_valid_data(self, from_translate=False, data_mode='single'):
        # Load validation data
        batch_size = 1 if from_translate else 64
        if from_translate:
            self.valid_ref_files = self.data['valid_trg']
            if isinstance(self.valid_ref_files, str):
                self.valid_ref_files = list([self.valid_ref_files])

            self.valid_iterator = WMTIterator(
                    batch_size=batch_size,
                    mask=False,
                    pklfile=self.data['valid_src'],
                    imgfile=self.data['valid_img'],
                    srcdict=self.src_dict, n_words_src=self.n_words_src,
                    mode=data_mode)
        else:
            # Just for loss computation
            self.valid_iterator = WMTIterator(
                    batch_size=self.batch_size,
                    pklfile=self.data['valid_src'],
                    imgfile=self.data['valid_img'],
                    trgdict=self.trg_dict, srcdict=self.src_dict,
                    n_words_trg=self.n_words_trg, n_words_src=self.n_words_src,
                    mode='single')

        self.valid_iterator.read()

    def init_params(self):
        params = OrderedDict()

        # embedding weights for encoder (source language)
        params['Wemb_enc'] = norm_weight(self.n_words_src, self.embedding_dim, scale=self.weight_init)

        # embedding weights for decoder (target language)
        params['Wemb_dec'] = norm_weight(self.n_words_trg, self.embedding_dim, scale=self.weight_init)

        # convfeats (512) to ctx dim (2000) for image modality
        params = get_new_layer('ff')[0](params, prefix='ff_img_adaptor', nin=self.conv_dim, nout=self.ctx_dim, scale=self.weight_init)

        #############################################
        # Source sentence encoder: bidirectional GRU
        #############################################
        # Forward and backward encoder parameters
        params = get_new_layer('gru')[0](params, prefix='text_encoder'  , nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init)
        params = get_new_layer('gru')[0](params, prefix='text_encoder_r', nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init)

        ##########
        # Decoder
        ##########
        # init_state computation from mean context: 2000 -> 1000 if rnn_dim == 1000
        params = get_new_layer('ff')[0](params, prefix='ff_text_state_init', nin=self.ctx_dim, nout=self.rnn_dim, scale=self.weight_init)

        # GRU cond decoder
        params = init_gru_decoder_multi(params, prefix='decoder_multi', nin=self.embedding_dim,
                                        dim=self.rnn_dim, dimctx=self.ctx_dim, scale=self.weight_init)

        # readout
        # NOTE: In the text NMT, we also have logit_prev that is applied onto emb_trg
        # NOTE: ortho= changes from text NMT to SAT. Need to experiment
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru', nin=self.rnn_dim     , nout=self.embedding_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx', nin=self.ctx_dim     , nout=self.embedding_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit'    , nin=self.embedding_dim , nout=self.n_words_trg, scale=self.weight_init)

        # Save initial parameters for debugging purposes
        self.initial_params = params

    def build(self):
        # Source sentences: n_timesteps, n_samples
        x = tensor.matrix('x', dtype=INT)
        x_mask = tensor.matrix('x_mask', dtype=FLOAT)

        # Image: 196 (n_annotations) x n_samples x 512 (ctxdim)
        x_img = tensor.tensor3('x_img', dtype=FLOAT)

        # Target sentences: n_timesteps, n_samples
        y = tensor.matrix('y', dtype=INT)
        y_mask = tensor.matrix('y_mask', dtype=FLOAT)

        # Some shorthands for dimensions
        n_samples       = x.shape[1]
        n_timesteps     = x.shape[0]
        n_timesteps_trg = y.shape[0]

        # Store tensors
        self.inputs = OrderedDict()
        self.inputs['x']        = x         # Source words
        self.inputs['x_mask']   = x_mask    # Source mask
        self.inputs['x_img']    = x_img     # Image features
        self.inputs['y']        = y         # Target labels
        self.inputs['y_mask']   = y_mask    # Target mask

        ###################
        # Source embeddings
        ###################
        # Fetch source embeddings. Result is: (n_timesteps x n_samples x embedding_dim)
        emb_enc = self.tparams['Wemb_enc'][x.flatten()].reshape([n_timesteps, n_samples, self.embedding_dim])
        # -> n_timesteps x n_samples x embedding_dim

        # Pass the source word vectors through the GRU RNN
        emb_enc_rnns = get_new_layer('gru')[1](self.tparams, emb_enc, prefix='text_encoder', mask=x_mask)
        # -> n_timesteps x n_samples x rnn_dim

        # word embedding for backward rnn (source)
        # for the backward rnn, we just need to invert x and x_mask
        xr      = x[::-1]
        xr_mask = x_mask[::-1]
        emb_enc_r = self.tparams['Wemb_enc'][xr.flatten()].reshape([n_timesteps, n_samples, self.embedding_dim])
        # -> n_timesteps x n_samples x embedding_dim
        # Pass the source word vectors in reverse through the GRU RNN
        emb_enc_rnns_r = get_new_layer('gru')[1](self.tparams, emb_enc_r, prefix='text_encoder_r', mask=xr_mask)
        # -> n_timesteps x n_samples x rnn_dim

        # Source context will be the concatenation of forward and backward rnns
        # leading to a vector of 2*rnn_dim for each timestep
        text_ctx = tensor.concatenate([emb_enc_rnns[0], emb_enc_rnns_r[0][::-1]], axis=emb_enc_rnns[0].ndim-1)
        # -> n_timesteps x n_samples x 2*rnn_dim

        # mean of the context (across time) will be used to initialize decoder rnn
        text_ctx_mean = (text_ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
        # -> n_samples x ctx_dim (2*rnn_dim)

        # initial decoder state computed from source context mean
        # NOTE: Can the two initializer be merged into one?
        text_init_state = get_new_layer('ff')[1](self.tparams, text_ctx_mean, prefix='ff_text_state_init', activ='tanh')
        # -> n_samples x rnn_dim (last dim shrinked down by this FF to rnn_dim)

        #######################
        # Source image features
        #######################

        # Project image features to ctx_dim
        img_ctx = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_adaptor', activ='linear')
        # -> 196 x n_samples x ctx_dim

        # initial decoder state learned from mean image context
        # NOTE: Can the two initializer be merged into one?
        #img_init_state = get_new_layer('ff')[1](self.tparams, img_ctx.mean(0), prefix='ff_img_state_init', activ='tanh')
        # -> n_samples x rnn_dim

        ####################
        # Target embeddings
        ####################

        # Fetch target embeddings. Result is: (n_trg_timesteps x n_samples x embedding_dim)
        emb_trg = self.tparams['Wemb_dec'][y.flatten()].reshape([y.shape[0], y.shape[1], self.embedding_dim])

        # Shift it to right to leave place for the <bos> placeholder
        # We ignore the last word <eos> as we don't condition on it at the end
        # to produce another word
        emb_trg_shifted = tensor.zeros_like(emb_trg)
        emb_trg_shifted = tensor.set_subtensor(emb_trg_shifted[1:], emb_trg[:-1])
        emb_trg = emb_trg_shifted

        ##########
        # GRU Cond
        ##########
        # decoder - pass through the decoder conditional gru with attention
        dec_mult = gru_decoder_multi(self.tparams, emb_trg,
                                     prefix='decoder_multi',
                                     input_mask=y_mask,
                                     ctx1=text_ctx, ctx1_mask=x_mask,
                                     ctx2=img_ctx,
                                     one_step=False,
                                     init_state=text_init_state) # NOTE: init_state only text

        # gru_cond returns hidden state, weighted sum of context vectors and attentional weights.
        h       = dec_mult[0]    # (n_timesteps_trg, batch_size, rnn_dim)
        sumctx  = dec_mult[1]    # (n_timesteps_trg, batch_size, ctx*.shape[-1] (2000, 2*rnn_dim))

        self.alphas  = list(dec_mult[2:])

        logit    = emb_trg
        logit   += get_new_layer('ff')[1](self.tparams, h, prefix='ff_logit_gru', activ='linear')
        logit   += get_new_layer('ff')[1](self.tparams, sumctx, prefix='ff_logit_ctx', activ='linear')

        # tanh over logit
        logit = tanh(logit)


        # embedding_dim -> n_words_trg
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

        self.f_log_probs = theano.function(self.inputs.values(), cost)

        return cost

    def build_sampler(self):
        x               = tensor.matrix('x', dtype=INT)
        n_timesteps     = x.shape[0]
        n_samples       = x.shape[1]

        ################
        # Image features
        ################
        # 196 x 512
        x_img           = tensor.matrix('x_img', dtype=FLOAT)
        # Convert to 196 x 2000 (2*rnn_dim)
        img_ctx         = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_adaptor', activ='linear')
        # Broadcast middle dimension to make it 196 x 1 x 2000
        img_ctx         = img_ctx[:, None, :]
        # Take the mean over the first dimension: 1 x 2000
        img_ctx_mean    = img_ctx.mean(0)
        # Give the mean to compute the initial state: 1 x 1000
        #img_init_state  = get_new_layer('ff')[1](self.tparams, img_ctx_mean, prefix='ff_img_state_init', activ='tanh')

        #####################
        # Text Bi-GRU Encoder
        #####################
        emb             = self.tparams['Wemb_enc'][x.flatten()]
        emb             = emb.reshape([n_timesteps, n_samples, self.embedding_dim])
        proj            = get_new_layer('gru')[1](self.tparams, emb, prefix='text_encoder')

        embr            = self.tparams['Wemb_enc'][x[::-1].flatten()]
        embr            = embr.reshape([n_timesteps, n_samples, self.embedding_dim])
        projr           = get_new_layer('gru')[1](self.tparams, embr, prefix='text_encoder_r')

        # concatenate forward and backward rnn hidden states
        text_ctx        = tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

        # get the input for decoder rnn initializer mlp
        text_ctx_mean   = text_ctx.mean(0)
        text_init_state = get_new_layer('ff')[1](self.tparams, text_ctx_mean, prefix='ff_text_state_init', activ='tanh')

        ################
        # Build f_init()
        ################
        inps            = [x, x_img]
        outs            = [text_init_state, text_ctx, img_ctx]
        self.f_init     = theano.function(inps, outs, name='f_init')

        ###################
        # Target Embeddings
        ###################
        y               = tensor.vector('y_sampler', dtype=INT)
        emb             = tensor.switch(y[:, None] < 0,
                                        tensor.alloc(0., 1, self.tparams['Wemb_dec'].shape[1]),
                                        self.tparams['Wemb_dec'][y])

        ##########
        # Text GRU
        ##########
        dec_mult = gru_decoder_multi(self.tparams, emb,
                                     prefix='decoder_multi',
                                     input_mask=None,
                                     ctx1=text_ctx, ctx1_mask=None,
                                     ctx2=img_ctx,
                                     one_step=True,
                                     init_state=text_init_state)
        h      = dec_mult[0]
        sumctx = dec_mult[1]
        alphas = tensor.concatenate(dec_mult[2:], axis=-1)

        ########
        # Fusion
        ########
        logit       = emb
        logit       += get_new_layer('ff')[1](self.tparams, sumctx, prefix='ff_logit_ctx', activ='linear')
        logit       += get_new_layer('ff')[1](self.tparams, h, prefix='ff_logit_gru', activ='linear')
        logit = tanh(logit)


        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')

        # compute the logsoftmax
        next_log_probs = tensor.nnet.logsoftmax(logit)

        # Sample from the softmax distribution
        next_probs = tensor.exp(next_log_probs)

        # NOTE: We never use sampling and it incurs performance penalty
        # let's disable it for now
        #next_word = self.trng.multinomial(pvals=next_probs).argmax(1)

        ################
        # Build f_next()
        ################
        inputs = [y, text_init_state, text_ctx, img_ctx]
        outs = [next_log_probs, h, alphas]
        self.f_next = theano.function(inputs, outs, name='f_next')

    def get_alpha_regularizer(self, alpha_c):
        alpha_c = theano.shared(np.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((1.-self.alphas[1].sum(0))**2).sum(0).mean()
        return alpha_reg
