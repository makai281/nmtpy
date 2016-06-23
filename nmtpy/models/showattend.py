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

        dicts = kwargs['dicts']
        assert 'trg' in dicts
        self.trg_dict, trg_idict = load_dictionary(dicts['trg'])
        self.n_words_trg = min(self.n_words_trg, len(self.trg_dict)) if self.n_words_trg > 0 else len(self.trg_dict)

        # Convolutional feature dim
        self.n_convfeats = 512

        # Collect options
        self.options = dict(self.__dict__)
        self.trg_idict = trg_idict

        self.set_nanguard()
        self.set_trng(seed)
        self.set_dropout(False)

    def load_data(self):
        self.train_iterator = WMTIterator(
                self.batch_size,
                self.data['train_src'],
                img_feats_file=self.data['train_img'],
                trg_dict=self.trg_dict,
                n_words_trg=self.n_words_trg, mode='pairs', shuffle=True)
        self.load_valid_data()

    def load_valid_data(self, from_translate=False):
        batch_size = 1 if from_translate else 64
        if from_translate:
            self.valid_ref_files = self.data['valid_trg']
            if isinstance(self.valid_ref_files, str):
                self.valid_ref_files = list([self.valid_ref_files])

            self.valid_iterator = WMTIterator(
                    batch_size, self.data['valid_src'],
                    img_feats_file=self.data['valid_img'],
                    mode='single')
        else:
            # Just for loss computation
            self.valid_iterator = WMTIterator(
                    batch_size, self.data['valid_src'],
                    img_feats_file=self.data['valid_img'],
                    trg_dict=self.trg_dict,
                    n_words_trg=self.n_words_trg,
                    mode='single')

    def init_params(self):
        params = OrderedDict()

        # embedding weights for decoder
        params['Wemb'] = norm_weight(self.n_words_trg, self.trg_emb_dim, scale=self.weight_init)

        # initial state initializer
        params = get_new_layer('ff')[0](params, prefix='ff_state', nin=self.n_convfeats, nout=self.rnn_dim, scale=self.weight_init)

        # decoder
        params = get_new_layer('gru_cond')[0](params, prefix='decoder', nin=self.trg_emb_dim, dim=self.rnn_dim, dimctx=self.n_convfeats, scale=self.weight_init)

        # readout
        # NOTE: First two are orthogonally initialized in arctic-captions
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru'   , nin=self.rnn_dim, nout=self.trg_emb_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx'   , nin=self.n_convfeats, nout=self.trg_emb_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit'       , nin=self.trg_emb_dim, nout=self.n_words_trg, scale=self.weight_init)

        self.initial_params = params

    def build(self):
        # Image: 196 (n_annotations) x n_samples x 512 (ctxdim)
        x_img = tensor.tensor3('x_img', dtype=FLOAT)

        # Target sentences: n_timesteps, n_samples
        y = tensor.matrix('y', dtype=INT)
        y_mask = tensor.matrix('y_mask', dtype=FLOAT)

        # Store tensors
        self.inputs['x_img'] = x_img
        self.inputs['y'] = y
        self.inputs['y_mask'] = y_mask

        # index into the word embedding matrix, shift it forward in time
        # to leave all-zeros in the first timestep and to ignore the last word
        # <eos> upon which we'll never condition
        emb = self.tparams['Wemb'][y.flatten()].reshape([y.shape[0], y.shape[1], self.trg_emb_dim])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        # Mean ctx vector
        # 1 x n_samples x 512
        ctx_mean = x_img.mean(0)

        # initial decoder state learned from mean context
        # 1 x n_samples x rnn_dim
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')

        # decoder - pass through the decoder conditional gru with attention
        proj = get_new_layer('gru_cond')[1](self.tparams, emb,
                                            prefix='decoder',
                                            mask=y_mask, context=x_img,
                                            context_mask=None,
                                            one_step=False,
                                            init_state=init_state)
        # gru_cond returns hidden state, weighted sum of context vectors and attentional weights.
        proj_h = proj[0]
        ctxs = proj[1]
        alphas = proj[2]

        # rnn_dim -> trg_emb_dim
        logit = get_new_layer('ff')[1](self.tparams, proj_h, prefix='ff_logit_gru', activ='linear')

        # prev2out == True in arctic-captions
        logit += emb

        # ctx2out == True in arctic-captions
        logit += get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')

        # tanh over logit
        logit = tanh(logit)

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

    def beam_search(self, inputs, beam_size=12, maxlen=50, suppress_unks=False, **kwargs):
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
