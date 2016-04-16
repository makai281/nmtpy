#!/usr/bin/env python
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
from ..typedef import *
from ..nmtutils import *
from ..iterators import WMTIterator

from ..models.basemodel import BaseModel

class Model(BaseModel):
    def __init__(self, seed, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(**kwargs)

        # We need both dictionaries
        dicts = kwargs['dicts']
        assert 'trg' in dicts and 'src' in dicts

        # We'll use both dictionaries
        self.src_dict, src_idict = load_dictionary(dicts['src'])
        self.trg_dict, trg_idict = load_dictionary(dicts['trg'])
        self.n_words_trg = min(self.n_words_trg, len(self.trg_dict)) if self.n_words_trg > 0 else len(self.trg_dict)
        self.n_words_src = min(self.n_words_src, len(self.src_dict)) if self.n_words_src > 0 else len(self.src_dict)

        # Convolutional feature dim
        self.n_convfeats = 512

        # Collect options
        self.options = dict(self.__dict__)

        # Set these here to not clutter options
        self.trg_idict = trg_idict
        self.src_idict = src_idict

        self.ctx_dim = 2 * self.rnn_dim
        self.set_nanguard()
        self.set_trng(seed)
        self.set_dropout(False)

    def load_data(self):
        # Load training data
        self.train_iterator = WMTIterator(
                self.batch_size,
                self.data['train_src'],
                img_feats_file=self.data['train_img'],
                trg_dict=self.trg_dict, src_dict=self.src_dict,
                n_words_trg=self.n_words_trg, n_words_src=self.n_words_src, shuffle=True)
        self.load_valid_data()

    def load_valid_data(self, from_translate=False):
        # Load validation data
        batch_size = 1 if from_translate else 64
        if from_translate:
            self.valid_ref_files = self.data['valid_trg']
            if isinstance(self.valid_ref_files, str):
                self.valid_ref_files = list([self.valid_ref_files])

            self.valid_iterator = WMTIterator(
                    batch_size, self.data['valid_src'],
                    img_feats_file=self.data['valid_img'],
                    src_dict=self.src_dict, n_words_src=self.n_words_src,
                    mode='single')
        else:
            # Just for loss computation
            self.valid_iterator = WMTIterator(
                    batch_size, self.data['valid_src'],
                    img_feats_file=self.data['valid_img'],
                    trg_dict=self.trg_dict, src_dict=self.src_dict,
                    n_words_trg=self.n_words_trg, n_words_src=self.n_words_src,
                    mode='single')

    def init_params(self):
        params = OrderedDict()

        # embedding weights for encoder (source language)
        params['Wemb_enc'] = norm_weight(self.n_words_src, self.src_emb_dim, scale=self.weight_init)

        # embedding weights for decoder (target language)
        params['Wemb_dec'] = norm_weight(self.n_words_trg, self.trg_emb_dim, scale=self.weight_init)

        # convfeats (512) to RNN dim (1000) for image modality
        params = get_new_layer('ff')[0](params, prefix='ff_img_adaptor', nin=self.n_convfeats, nout=self.rnn_dim, scale=self.weight_init)

        #############################################
        # Source sentence encoder: bidirectional GRU
        #############################################
        # Forward and backward encoder parameters
        params = get_new_layer('gru')[0](params, prefix='text_encoder'  , nin=self.src_emb_dim, dim=self.rnn_dim, scale=self.weight_init)
        params = get_new_layer('gru')[0](params, prefix='text_encoder_r', nin=self.src_emb_dim, dim=self.rnn_dim, scale=self.weight_init)

        ##########
        # Decoder
        ##########
        # init_state computation from mean context: 2000 -> 1000 if rnn_dim == 1000
        params = get_new_layer('ff')[0](params, prefix='ff_text_state_init', nin=self.ctx_dim, nout=self.rnn_dim, scale=self.weight_init)

        # GRU Decoder for text path
        params = get_new_layer('gru_cond')[0](params, prefix='decoder_text', nin=self.trg_emb_dim, dim=self.rnn_dim, dimctx=self.ctx_dim, scale=self.weight_init)

        ###############
        # Image Decoder
        ###############
        # init_state computation from mean image context: 1000 -> 1000
        params = get_new_layer('ff')[0](params, prefix='ff_img_state_init', nin=self.rnn_dim, nout=self.rnn_dim, scale=self.weight_init)

        # GRU decoder for image path (receives target embeddings as state_below)
        params = get_new_layer('gru_cond')[0](params, prefix='decoder_img', nin=self.trg_emb_dim, dim=self.rnn_dim, dimctx=self.rnn_dim, scale=self.weight_init)

        # readout
        # NOTE: In the text NMT, we also have logit_prev that is applied onto emb_trg
        # NOTE: ortho= changes from text NMT to SAT. Need to experiment
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru', nin=self.rnn_dim     , nout=self.trg_emb_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx', nin=self.rnn_dim     , nout=self.trg_emb_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit'    , nin=self.trg_emb_dim , nout=self.n_words_trg, scale=self.weight_init)

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
        self.inputs['x']        = x         # Source words
        self.inputs['x_mask']   = x_mask    # Source mask
        self.inputs['x_img']    = x_img     # Image features
        self.inputs['y']        = y         # Target labels
        self.inputs['y_mask']   = y_mask    # Target mask

        # Project image features to rnn_dim
        img_ctx = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img_adaptor', activ='linear')
        # -> 196 x n_samples x rnn_dim

        ###################
        # Source embeddings
        ###################
        # Fetch source embeddings. Result is: (n_timesteps x n_samples x src_emb_dim)
        emb_enc = self.tparams['Wemb_enc'][x.flatten()].reshape([n_timesteps, n_samples, self.src_emb_dim])
        # -> n_timesteps x n_samples x src_emb_dim

        # Pass the source word vectors through the GRU RNN
        emb_enc_rnns = get_new_layer('gru')[1](self.tparams, emb_enc, prefix='text_encoder', mask=x_mask,
                                               profile=self.profile, mode=self.func_mode)
        # -> n_timesteps x n_samples x rnn_dim

        # word embedding for backward rnn (source)
        # for the backward rnn, we just need to invert x and x_mask
        xr      = x[::-1]
        xr_mask = x_mask[::-1]
        emb_enc_r = self.tparams['Wemb_enc'][xr.flatten()].reshape([n_timesteps, n_samples, self.src_emb_dim])
        # -> n_timesteps x n_samples x src_emb_dim
        # Pass the source word vectors in reverse through the GRU RNN
        emb_enc_rnns_r = get_new_layer('gru')[1](self.tparams, emb_enc_r, prefix='text_encoder_r', mask=xr_mask,
                                                 profile=self.profile, mode=self.func_mode)
        # -> n_timesteps x n_samples x rnn_dim

        # Source context will be the concatenation of forward and backward rnns
        # leading to a vector of 2*rnn_dim for each timestep
        text_ctx = tensor.concatenate([emb_enc_rnns[0], emb_enc_rnns_r[0][::-1]], axis=emb_enc_rnns[0].ndim-1)
        # -> n_timesteps x n_samples x 2*rnn_dim

        # mean of the context (across time) will be used to initialize decoder rnn
        text_ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
        # -> n_samples x ctx_dim (2*rnn_dim)

        # initial decoder state computed from source context mean
        # NOTE: Can the two initializer be merged into one?
        text_init_state = get_new_layer('ff')[1](self.tparams, text_ctx_mean, prefix='ff_text_state_init', activ='tanh')
        # -> n_samples x rnn_dim (last dim shrinked down by this FF to rnn_dim)

        #######################
        # Source image features
        #######################

        # initial decoder state learned from mean image context
        # NOTE: Can the two initializer be merged into one?
        img_init_state = get_new_layer('ff')[1](self.tparams, img_ctx.mean(0), prefix='ff_img_state_init', activ='tanh')
        # -> n_samples x rnn_dim

        ####################
        # Target embeddings
        ####################

        # Fetch target embeddings. Result is: (n_trg_timesteps x n_samples x trg_emb_dim)
        emb_trg = self.tparams['Wemb_dec'][y.flatten()].reshape([y.shape[0], y.shape[1], self.trg_emb_dim])

        # Shift it to right to leave place for the <bos> placeholder
        # We ignore the last word <eos> as we don't condition on it at the end
        # to produce another word
        emb_trg_shifted = tensor.zeros_like(emb_trg)
        emb_trg_shifted = tensor.set_subtensor(emb_trg_shifted[1:], emb_trg[:-1])
        emb_trg = emb_trg_shifted

        ###########
        # Image GRU
        ###########
        # decoder - pass through the decoder conditional gru with attention
        img_rnn = get_new_layer('gru_cond')[1](self.tparams, emb_trg,
                                               prefix='decoder_img',
                                               mask=y_mask, context=x_img,
                                               context_mask=None,
                                               one_step=False,
                                               init_state=img_init_state,
                                               profile=self.profile,
                                               model=self.func_mode)

        # gru_cond returns hidden state, weighted sum of context vectors and attentional weights.
        img_h       = img_rnn[0]
        img_sumctx  = img_rnn[1]
        img_alphas  = img_rnn[2]

        ##########
        # Text GRU
        ##########
        text_rnn = get_new_layer('gru_cond')[1](self.tparams, emb_trg,
                                                prefix='decoder_text',
                                                mask=y_mask, context=text_ctx, # word contexts, e.g. n_timesteps x n_samples x 2*rnn_dim
                                                context_mask=x_mask,
                                                one_step=False,
                                                init_state=init_state,
                                                profile=self.profile,
                                                mode=self.func_mode)
        text_h      = text_rnn[0]
        text_sumctx = text_rnn[1]
        text_alphas = text_rnn[2]


        # Sum hidden states of modalities
        # rnn_dim -> trg_emb_dim
        logit = get_new_layer('ff')[1](self.tparams, (img_h + text_h), prefix='ff_logit_gru', activ='linear')

        # prev2out == True in arctic-captions
        logit += emb_trg

        # ctx2out == True in arctic-captions
        logit += get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')

        # tanh over logit
        logit = tanh(logit)

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

        # trg_emb_dim -> n_words_trg
        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_img_logit', activ='linear')
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

        return cost.mean()

    def build_sampler(self):
        # context is the convolutional vectors themselves
        # 196 x 512
        ctx = tensor.matrix('x_img', dtype=FLOAT)

        # 1 x 512
        ctx_mean = ctx.mean(0)

        # initial decoder state
        # (probably) 1 x rnn_dim
        # Can be encapsulated with list to support multiple RNN layers in future
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')

        # Takes image annotation vectors and returns
        # it with the initial state of GRU
        self.f_init = theano.function([ctx], [init_state[None, :], ctx], name='f_init', profile=self.profile)

        y = tensor.vector('y_sampler', dtype=INT)
        init_state = tensor.matrix('init_state', dtype=FLOAT)

        # if it's the first word, emb should be all zero and it is indicated by
        # beam search who sends -1 for the initial word
        # n_words x emb_dim when y != -1
        emb = tensor.switch(y[:, None] < 0,
                            tensor.alloc(0., 1, self.trg_emb_dim),
                            self.tparams['Wemb'][y])

        # apply one step of conditional gru with attention
        r = get_new_layer('gru_cond')[1](self.tparams, emb,
                                            prefix='decoder',
                                            mask=None, context=ctx[:, None, :],
                                            one_step=True,
                                            init_state=init_state)
        # get the next hidden state
        # get the weighted average of context for this target word y
        next_state = r[0]

        # 1 x 512
        ctxs = r[1]

        logit  = emb
        logit += get_new_layer('ff')[1](self.tparams, next_state, prefix='ff_logit_gru', activ='linear')
        logit += get_new_layer('ff')[1](self.tparams, ctxs      , prefix='ff_logit_ctx', activ='linear')
        logit  = tanh(logit)

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')

        # compute the logsoftmax
        next_log_probs = tensor.nnet.logsoftmax(logit)

        # Sample from the softmax distribution
        next_probs = tensor.exp(next_log_probs)
        next_word = self.trng.multinomial(pvals=next_probs).argmax(1)

        # compile a function to do the whole thing above
        # sampled word for the next target, next hidden state to be used
        inputs = [y, ctx, init_state]
        outs = [next_log_probs, next_word, next_state]
        self.f_next = theano.function(inputs, outs, name='f_next', profile=self.profile)

    def beam_search(self, inputs, beam_size=12, maxlen=50):
        # Final results and their scores
        final_sample = []
        final_score  = []

        live_beam = 1
        dead_beam = 0

        # Initially we have one empty hypothesis with a score of 0
        hyp_states  = []
        hyp_samples = [[]]
        hyp_scores  = np.zeros(1).astype(FLOAT)

        # We only have single input which is ctx/x_img
        # We obtain the same ctx as ctx0 as well as the next_state
        # computed by the MLP ff_state
        # next_state: 1 x 1000
        # ctx0: 196 x 512
        next_state, ctx0 = self.f_init(*inputs)

        # Beginning-of-sentence indicator is -1
        next_w = np.array([-1], dtype=INT)

        for ii in xrange(maxlen):
            # Get next states
            # In the first iteration, we provide -1 and obtain the log_p's for the
            # first word. In the following iterations tiled_ctx becomes a batch
            # of duplicated left hypotheses. tiled_ctx is always the same except
            # the 2nd dimension as the context vectors of the source sequence
            # is always the same regardless of the decoding step.
            inps = [next_w, ctx0, next_state]
            next_log_p, _, next_state = self.f_next(*inps)

            # Compute sum of log_p's for the current n-gram hypotheses and flatten them
            cand_scores = hyp_scores[:, None] - next_log_p
            cand_flat = cand_scores.flatten()

            # Take the best beam_size-dead_beam hypotheses
            ranks_flat = cand_flat.argsort()[:(beam_size-dead_beam)]

            # Get their costs
            costs = cand_flat[ranks_flat]

            # Find out to which initial hypothesis idx this was belonging
            trans_indices = ranks_flat / self.n_words_trg
            # Find out the idx of the appended word
            word_indices = ranks_flat % self.n_words_trg

            # New states, scores and samples
            new_hyp_states  = []
            new_hyp_samples = []
            new_hyp_scores  = np.zeros(beam_size-dead_beam).astype(FLOAT)

            # Iterate over the hypotheses and add them to new_* lists
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_beam = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    # EOS detected
                    final_sample.append(new_hyp_samples[idx])
                    final_score.append(new_hyp_scores[idx])
                    dead_beam += 1
                else:
                    new_live_beam += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])

            hyp_scores = np.array(hyp_scores)
            live_beam = new_live_beam

            if new_live_beam < 1 or dead_beam >= beam_size:
                break

            # Take the idxs of each hyp's last word
            next_w = np.array([w[-1] for w in hyp_samples])
            next_state = np.array(hyp_states)

        # dump every remaining hypotheses
        if live_beam > 0:
            for idx in xrange(live_beam):
                final_sample.append(hyp_samples[idx])
                final_score.append(hyp_scores[idx])

        return final_sample, final_score
