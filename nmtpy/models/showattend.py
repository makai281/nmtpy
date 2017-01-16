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

from .basemodel import BaseModel

class Model(BaseModel):
    def __init__(self, seed, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(**kwargs)

        dicts = kwargs['dicts']
        assert 'trg' in dicts
        self.trg_dict, trg_idict = load_dictionary(dicts['trg'])
        self.n_words_trg = min(self.n_words_trg, len(self.trg_dict)) if self.n_words_trg > 0 else len(self.trg_dict)

        # Collect options
        self.set_options(self.__dict__)
        self.trg_idict = trg_idict

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

    def load_valid_data(self, from_translate=False, data_mode='single'):
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
        params['Wemb_enc'] = norm_weight(self.n_words_trg, self.trg_emb_dim, scale=self.weight_init)

        # initial state initializer
        params = get_new_layer('ff')[0](params, prefix='ff_state', nin=self.conv_dim, nout=self.rnn_dim, scale=self.weight_init)

        # decoder
        params = get_new_layer('gru_cond')[0](params, prefix='decoder', nin=self.trg_emb_dim, dim=self.rnn_dim, dimctx=self.conv_dim, scale=self.weight_init)

        # readout
        # NOTE: First two are orthogonally initialized in arctic-captions
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru'   , nin=self.rnn_dim, nout=self.trg_emb_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx'   , nin=self.conv_dim, nout=self.trg_emb_dim, scale=self.weight_init)
        params = get_new_layer('ff')[0](params, prefix='ff_logit'       , nin=self.trg_emb_dim, nout=self.n_words_trg, scale=self.weight_init)

        self.initial_params = params

    def build(self):
        # Image: 196 (n_annotations) x n_samples x conv_dim (ctxdim)
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
        emb = self.tparams['Wemb_enc'][y.flatten()].reshape([y.shape[0], y.shape[1], self.trg_emb_dim])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        # Mean ctx vector
        # 1 x n_samples x conv_dim
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
        self.alphas = proj[2]

        # rnn_dim -> trg_emb_dim
        logit = get_new_layer('ff')[1](self.tparams, proj_h, prefix='ff_logit_gru', activ='linear')

        # prev2out == True in arctic-captions
        logit += emb

        # ctx2out == True in arctic-captions
        logit += get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')

        # tanh over logit
        logit = tanh(logit)


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

        self.f_log_probs = theano.function(self.inputs.values(), cost)

        return cost.mean()

    def build_sampler(self):
        # context is the convolutional vectors themselves
        # 196 x conv_dim
        ctx = tensor.matrix('x_img', dtype=FLOAT)

        # 1 x conv_dim
        ctx_mean = ctx.mean(0)

        # initial decoder state
        # (probably) 1 x rnn_dim
        # Can be encapsulated with list to support multiple RNN layers in future
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')

        # Takes image annotation vectors and returns
        # it with the initial state of GRU
        self.f_init = theano.function([ctx], [init_state[None, :], ctx], name='f_init')

        y = tensor.vector('y_sampler', dtype=INT)
        init_state = tensor.matrix('init_state', dtype=FLOAT)

        # if it's the first word, emb should be all zero and it is indicated by
        # beam search who sends -1 for the initial word
        # n_words x emb_dim when y != -1
        emb = tensor.switch(y[:, None] < 0,
                            tensor.alloc(0., 1, self.trg_emb_dim),
                            self.tparams['Wemb_enc'][y])

        # apply one step of conditional gru with attention
        r = get_new_layer('gru_cond')[1](self.tparams, emb,
                                            prefix='decoder',
                                            mask=None, context=ctx[:, None, :],
                                            one_step=True,
                                            init_state=init_state)
        # get the next hidden state
        # get the weighted average of context for this target word y
        next_state = r[0]

        # 1 x conv_dim
        ctxs = r[1]
        alphas = r[2]

        logit  = emb
        logit += get_new_layer('ff')[1](self.tparams, next_state, prefix='ff_logit_gru', activ='linear')
        logit += get_new_layer('ff')[1](self.tparams, ctxs      , prefix='ff_logit_ctx', activ='linear')
        logit  = tanh(logit)


        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')

        # compute the logsoftmax
        next_log_probs = tensor.nnet.logsoftmax(logit)

        # Sample from the softmax distribution
        next_probs = tensor.exp(next_log_probs)
        next_word = self.trng.multinomial(pvals=next_probs).argmax(1)

        # compile a function to do the whole thing above
        # sampled word for the next target, next hidden state to be used
        inputs = [y, ctx, init_state]
        outs = [next_log_probs, next_word, next_state, alphas]
        self.f_next = theano.function(inputs, outs, name='f_next')

    def beam_search(self, inputs, beam_size=12, maxlen=50, suppress_unks=False, **kwargs):
        get_att = kwargs.get('get_att_alphas', False)

        # Final results and their scores
        final_sample        = []
        final_score         = []
        final_alignments    = []

        # Initially we have one empty hypothesis with a score of 0
        hyp_alignments  = [[]]
        hyp_samples     = [[]]
        hyp_scores      = np.zeros(1, dtype=FLOAT)

        # We only have single input which is ctx/x_img
        # We obtain the same ctx as ctx0 as well as the next_state
        # computed by the MLP ff_state
        # next_state: 1 x 1000
        # ctx0: 196 x conv_dim
        next_state, ctx0 = self.f_init(*inputs)

        # Beginning-of-sentence indicator is -1
        next_w = -1 * np.ones((1,), dtype=INT)

        live_beam = beam_size

        for ii in xrange(maxlen):
            inps = [next_w, ctx0, next_state]
            next_log_p, _, next_state, alphas = self.f_next(*inps)

            # For each f_next, we obtain a new set of alpha's for the next_w
            # for each hypothesis in the beam search

            if suppress_unks:
                next_log_p[:, 1] = -np.inf

            # Compute sum of log_p's for the current hypotheses
            cand_scores = hyp_scores[:, None] - next_log_p

            # Flatten by modifying .shape (faster)
            cand_scores.shape = cand_scores.size

            # Take the best live_beam hypotheses
            # argpartition makes a partial sort which is faster than argsort
            # (Idea taken from https://github.com/rsennrich/nematus)
            ranks_flat = cand_scores.argpartition(live_beam-1)[:live_beam]

            # Get the costs
            costs = cand_scores[ranks_flat]

            # New states, scores and samples
            live_beam           = 0
            new_hyp_scores      = []
            new_hyp_samples     = []
            new_hyp_alignments  = []

            # This will be the new next states in the next iteration
            hyp_states          = []

            # Find out to which initial hypothesis idx this was belonging
            # Find out the idx of the appended word
            trans_idxs  = ranks_flat / self.n_words_trg
            word_idxs   = ranks_flat % self.n_words_trg
            # Iterate over the hypotheses and add them to new_* lists
            for idx, [ti, wi] in enumerate(zip(trans_idxs, word_idxs)):
                # Form the new hypothesis by appending new word to the left hyp
                new_hyp = hyp_samples[ti] + [wi]
                new_ali = hyp_alignments[ti] + [alphas[ti]]

                if wi == 0:
                    # <eos> found, separate out finished hypotheses
                    final_sample.append(new_hyp)
                    final_score.append(costs[idx])
                    final_alignments.append(new_ali)
                else:
                    # Add formed hypothesis to the new hypotheses list
                    new_hyp_samples.append(new_hyp)
                    # Cumulated cost of this hypothesis
                    new_hyp_scores.append(costs[idx])
                    # Hidden state of the decoder for this hypothesis
                    hyp_states.append(next_state[ti])
                    new_hyp_alignments.append(new_ali)
                    live_beam += 1

            hyp_scores  = np.array(new_hyp_scores, dtype=FLOAT)
            hyp_samples = new_hyp_samples
            hyp_alignments = new_hyp_alignments

            if live_beam == 0:
                break

            # Take the idxs of each hyp's last word
            next_w      = np.array([w[-1] for w in hyp_samples])
            next_state  = np.array(hyp_states, dtype=FLOAT)

        # dump every remaining hypotheses
        for idx in xrange(live_beam):
            final_sample.append(hyp_samples[idx])
            final_score.append(hyp_scores[idx])
            final_alignments.append(hyp_alignments[idx])

        if get_att:
            return final_sample, final_score, final_alignments
        else:
            return final_sample, final_score
