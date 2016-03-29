from six.moves import range
from six.moves import zip

# Python
import os
import copy
import cPickle
import inspect
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
            if 'src' in dicts:
                self.src_dict, src_idict = load_dictionary(dicts['src'])
                self.n_words_src = min(self.n_words_src, len(self.src_dict)) if self.n_words_src > 0 else len(self.src_dict)
            if 'trg' in dicts:
                self.trg_dict, trg_idict = load_dictionary(dicts['trg'])
                self.n_words_trg = min(self.n_words_trg, len(self.trg_dict)) if self.n_words_trg > 0 else len(self.trg_dict)

        self.options = dict(self.__dict__)
        self.trg_idict = trg_idict
        self.src_idict = src_idict

        self.ctx_dim = 2 * self.rnn_dim
        self.set_nanguard()
        self.set_trng(seed)

        # Set iterator types here
        self.train_iter = "bitext"
        self.valid_src_iter = "bitext"
        self.valid_trg_iter = "bitext"

    def load_data(self):
        self.train_iterator = get_iterator(self.train_iter)(
                                self.data['train_src'], self.src_dict,
                                self.data['train_trg'], self.trg_dict,
                                batch_size=self.batch_size,
                                n_words_src=self.n_words_src,
                                n_words_trg=self.n_words_trg)

        # Prepare batches
        self.train_iterator.prepare_batches()

        # Validation
        valid_trg_files = self.data['valid_trg']
        if isinstance(valid_trg_files, str):
            valid_trg_files = list([valid_trg_files])

        # Take the first validation item for NLL computation
        self.valid_iterator = get_iterator(self.valid_src_iter)(
                                self.data['valid_src'], self.src_dict,
                                valid_trg_files[0], self.trg_dict, batch_size=64,
                                n_words_src=self.n_words_src, n_words_trg=self.n_words_trg)
        self.valid_iterator.prepare_batches()

    def init_params(self):
        params = OrderedDict()

        # embedding weights for encoder and decoder
        params['Wemb_enc'] = norm_weight(self.n_words_src, self.embedding_dim)
        params['Wemb_dec'] = norm_weight(self.n_words_trg, self.embedding_dim)

        # encoder: bidirectional RNN
        #########
        # Forward encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder', nin=self.embedding_dim, dim=self.rnn_dim)
        # Backwards encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder_r', nin=self.embedding_dim, dim=self.rnn_dim)

        # Context is the concatenation of forward and backwards encoder

        # init_state, init_cell
        params = get_new_layer('ff')[0](params, prefix='ff_state', nin=self.ctx_dim, nout=self.rnn_dim)
        # decoder
        params = get_new_layer(self.dec_type)[0](params, prefix='decoder', nin=self.embedding_dim, dim=self.rnn_dim, dimctx=self.ctx_dim)

        # readout
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru'   , nin=self.rnn_dim, nout=self.embedding_dim, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_prev'  , nin=self.embedding_dim, nout=self.embedding_dim, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx'   , nin=self.ctx_dim, nout=self.embedding_dim, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit'       , nin=self.embedding_dim, nout=self.n_words_trg)

        self.initial_params = params

    def build(self):
        # description string: #words x #samples
        x = tensor.matrix('x', dtype=INT)
        x_mask = tensor.matrix('x_mask', dtype=FLOAT)
        y = tensor.matrix('y', dtype=INT)
        y_mask = tensor.matrix('y_mask', dtype=FLOAT)

        self.inputs['x'] = x
        self.inputs['x_mask'] = x_mask
        self.inputs['y'] = y
        self.inputs['y_mask'] = y_mask

        # for the backward rnn, we just need to invert x and x_mask
        xr = x[::-1]
        xr_mask = x_mask[::-1]

        n_timesteps = x.shape[0]
        n_timesteps_trg = y.shape[0]
        n_samples = x.shape[1]

        # word embedding for forward rnn (source)
        emb = self.tparams['Wemb_enc'][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.embedding_dim])
        proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder', mask=x_mask,
                                               profile=self.profile, mode=self.func_mode)

        # word embedding for backward rnn (source)
        embr = self.tparams['Wemb_enc'][xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r', mask=xr_mask,
                                                profile=self.profile, mode=self.func_mode)

        # context will be the concatenation of forward and backward rnns
        ctx = tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

        # mean of the context (across time) will be used to initialize decoder rnn
        ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

        # NOTE: Tried this, no improvement
        # or you can use the last state of forward + backward encoder rnns
        # ctx_mean = tensor.concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

        # initial decoder state
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')

        # word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        emb = self.tparams['Wemb_dec'][y.flatten()]
        emb = emb.reshape([n_timesteps_trg, n_samples, self.embedding_dim])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        # decoder - pass through the decoder conditional gru with attention
        proj = get_new_layer(self.dec_type)[1](self.tparams, emb,
                                                    prefix='decoder',
                                                    mask=y_mask, context=ctx,
                                                    context_mask=x_mask,
                                                    one_step=False,
                                                    init_state=init_state,
                                                    profile=self.profile,
                                                    mode=self.func_mode)
        # hidden states of the decoder gru
        proj_h = proj[0]

        # weighted averages of context, generated by attention module
        ctxs = proj[1]

        # weights (alignment matrix)
        alphas = proj[2]

        # compute word probabilities
        logit_gru = get_new_layer('ff')[1](self.tparams, proj_h, prefix='ff_logit_gru', activ='linear')
        logit_prev = get_new_layer('ff')[1](self.tparams, emb, prefix='ff_logit_prev', activ='linear')
        logit_ctx = get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')

        logit = tanh(logit_gru + logit_prev + logit_ctx)

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

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

        # For alpha regularization
        self.x_mask = x_mask
        self.y_mask = y_mask
        self.alphas = alphas

        return cost.mean(), norm_cost.mean()

    def add_alpha_regularizer(self, cost, alpha_c):
        alpha_c = theano.shared(alpha_c.astype(FLOAT), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(self.y_mask.sum(0) // self.x_mask.sum(0), FLOAT)[:, None] -
             self.alphas.sum(0))**2).sum(1).mean()
        cost += alpha_reg
        return cost

    def build_sampler(self):
        x = tensor.matrix('x', dtype=INT)
        xr = x[::-1]
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # word embedding (source), forward and backward
        emb = self.tparams['Wemb_enc'][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.embedding_dim])

        embr = self.tparams['Wemb_enc'][xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])

        # encoder
        proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder')
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r')

        # concatenate forward and backward rnn hidden states
        ctx = tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

        # get the input for decoder rnn initializer mlp
        ctx_mean = ctx.mean(0)
        # ctx_mean = tensor.concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')

        outs = [init_state, ctx]
        self.f_init = theano.function([x], outs, name='f_init', profile=self.profile)

        # x: 1 x 1
        y = tensor.vector('y_sampler', dtype=INT)
        init_state = tensor.matrix('init_state', dtype=FLOAT)

        # if it's the first word, emb should be all zero and it is indicated by -1
        emb = tensor.switch(y[:, None] < 0,
                            tensor.alloc(0., 1, self.tparams['Wemb_dec'].shape[1]),
                            self.tparams['Wemb_dec'][y])

        # apply one step of conditional gru with attention
        # get the next hidden state
        # get the weighted averages of context for this target word y
        r = get_new_layer(self.dec_type)[1](self.tparams, emb,
                                                    prefix='decoder',
                                                    mask=None, context=ctx,
                                                    one_step=True,
                                                    init_state=init_state)

        next_state = r[0]
        ctxs = r[1]

        logit_prev = get_new_layer('ff')[1](self.tparams, emb,          prefix='ff_logit_prev',activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, ctxs,         prefix='ff_logit_ctx', activ='linear')
        logit_gru  = get_new_layer('ff')[1](self.tparams, next_state,   prefix='ff_logit_gru', activ='linear')

        logit = tanh(logit_gru + logit_prev + logit_ctx)

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

        logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')

        # compute the logsoftmax
        next_log_probs = tensor.nnet.logsoftmax(logit)

        # Sample from the softmax distribution
        next_probs = tensor.exp(next_log_probs)
        next_word = self.trng.multinomial(pvals=next_probs).argmax(1)

        # compile a function to do the whole thing above
        # next hidden state to be used
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

        # get initial state of decoder rnn and encoder context vectors
        # ctx0: the set of context vectors leading to the next_state
        # with a shape of (n_words x 1 x ctx_dim)
        # next_state: mean context vector (ctx0.mean()) passed through FF with a final
        # shape of (1 x 1 x ctx_dim)
        next_state, ctx0 = self.f_init(inputs[0])

        # Beginning-of-sentence indicator is -1
        next_w = -1 * np.ones((1,)).astype(INT)

        # maxlen or 3 times source length
        maxlen = min(maxlen, inputs[0].shape[0] * 3)

        for ii in xrange(maxlen):
            # Always starts with the initial tstep's context vectors
            # e.g. we have a ctx0 of shape (n_words x 1 x ctx_dim)
            # Tiling it live_beam times makes it (n_words x live_beam x ctx_dim)
            # thus we create sth like a batch of live_beam size with every word duplicated
            # for further state expansion.
            tiled_ctx = np.tile(ctx0, [live_beam, 1])

            # Get next states
            # In the first iteration, we provide -1 and obtain the log_p's for the
            # first word. In the following iterations tiled_ctx becomes a batch
            # of duplicated left hypotheses. tiled_ctx is always the same except
            # the 2nd dimension as the context vectors of the source sequence
            # is always the same regardless of the decoding step.
            next_log_p, _, next_state = self.f_next(*[next_w, tiled_ctx, next_state])

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
