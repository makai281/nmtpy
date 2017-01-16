from six.moves import range
from six.moves import zip

# Python
import os
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
from ..iterators.text import TextIterator
from ..iterators.bitext import BiTextIterator
from .basemodel import BaseModel

class Model(BaseModel):
    def __init__(self, seed, logger, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(**kwargs)

        # Load dictionaries
        dicts = kwargs['dicts']

        # Use GRU by default as encoder
        self.enc_type = kwargs.get('enc_type', 'gru')

        # Do we apply layer normalization to GRU?
        self.lnorm = kwargs.get('layer_norm', True)

        # Shuffle mode (default: No shuffle)
        self.smode = kwargs.get('shuffle_mode', 'trglen')

        # Get dropout parameters
        # Let's keep the defaults as 0 to not use dropout
        # You can adjust those from your conf files.
        self.emb_dropout = kwargs.get('emb_dropout', 0.2)
        self.ctx_dropout = kwargs.get('ctx_dropout', 0.4)
        self.out_dropout = kwargs.get('out_dropout', 0.4)

        # Number of additional GRU encoders for source sentences
        self.n_enc_layers  = kwargs.get('n_enc_layers' , 1)

        # Use a single embedding matrix for target words?
        self.tied_trg_emb = kwargs.get('tied_trg_emb', True)

        # Load dictionaries, limit shortlist size if requested
        self.src_dict, src_idict = load_dictionary(dicts['src'])
        self.n_words_src = min(self.n_words_src, len(self.src_dict)) \
                if self.n_words_src > 0 else len(self.src_dict)
        self.trg_dict, trg_idict = load_dictionary(dicts['trg'])
        self.n_words_trg = min(self.n_words_trg, len(self.trg_dict)) \
                if self.n_words_trg > 0 else len(self.trg_dict)

        # Create options. This will saved as .pkl
        self.set_options(self.__dict__)

        self.trg_idict = trg_idict
        self.src_idict = src_idict

        # Context dimensionality is 2 times RNN since we use Bi-RNN
        self.ctx_dim = 2 * self.rnn_dim

        # Set the seed of Theano RNG
        self.set_trng(seed)

        # We call this once to setup dropout mechanism correctly
        self.set_dropout(False)
        self.logger = logger

    def info(self):
        self.logger.info('Source vocabulary size: %d', self.n_words_src)
        self.logger.info('Target vocabulary size: %d', self.n_words_trg)
        self.logger.info('%d training samples' % self.train_iterator.n_samples)
        self.logger.info('%d validation samples' % self.valid_iterator.n_samples)
        self.logger.info('dropout (emb,ctx,out): %.2f, %.2f, %.2f' % (self.emb_dropout, self.ctx_dropout, self.out_dropout))

    def load_valid_data(self, from_translate=False):
        self.valid_ref_files = self.data['valid_trg']
        if isinstance(self.valid_ref_files, str):
            self.valid_ref_files = list([self.valid_ref_files])

        if from_translate:
            self.valid_iterator = TextIterator(
                                    mask=False,
                                    batch_size=1,
                                    file=self.data['valid_src'], dict=self.src_dict,
                                    n_words=self.n_words_src)
        else:
            # Take the first validation item for NLL computation
            self.valid_iterator = BiTextIterator(
                                    batch_size=self.batch_size,
                                    srcfile=self.data['valid_src'], srcdict=self.src_dict,
                                    trgfile=self.valid_ref_files[0], trgdict=self.trg_dict,
                                    n_words_src=self.n_words_src, n_words_trg=self.n_words_trg)

        self.valid_iterator.read()

    def load_data(self):
        self.train_iterator = BiTextIterator(
                                batch_size=self.batch_size,
                                shuffle_mode=self.smode,
                                logger=self.logger,
                                srcfile=self.data['train_src'], srcdict=self.src_dict,
                                trgfile=self.data['train_trg'], trgdict=self.trg_dict,
                                n_words_src=self.n_words_src,
                                n_words_trg=self.n_words_trg)

        # Prepare batches
        self.train_iterator.read()
        self.load_valid_data()

    def add_alpha_regularizer(self, alpha_c):
        alpha_c = theano.shared(np.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(self.y_mask.sum(0) // self.x_mask.sum(0), FLOAT)[:, None] -
             self.alphas.sum(0))**2).sum(1).mean()
        return alpha_reg

    def beam_search(self, inputs, beam_size=12, maxlen=50, suppress_unks=False, **kwargs):
        # Final results and their scores
        final_sample        = []
        final_score         = []
        final_alignments    = []

        # Initially we have one empty hypothesis with a score of 0
        hyp_alignments  = [[]]
        hyp_samples     = [[]]
        hyp_scores      = np.zeros(1, dtype=FLOAT)

        # get initial state of decoder rnn and encoder context vectors
        # ctx0: the set of context vectors leading to the next_state
        # with an initial shape of (n_src_words x 1 x ctx_dim)

        # next_state: mean context vector (ctx0.mean()) passed through FF with a final
        # shape of (1 x rnn_dim)
        next_state, ctx0 = self.f_init(inputs[0])

        # Beginning-of-sentence indicator is -1
        next_w = -1 * np.ones((1,), dtype=INT)

        # maxlen or 3 times source length
        maxlen = min(maxlen, inputs[0].shape[0] * 3)

        # Always starts with the initial tstep's context vectors
        # e.g. we have a ctx0 of shape (n_words x 1 x ctx_dim)
        # Tiling it live_beam times makes it (n_words x live_beam x ctx_dim)
        # thus we create sth like a batch of live_beam size with every word duplicated
        # for further state expansion.
        tiled_ctx = np.tile(ctx0, [1, 1])
        live_beam = beam_size

        for ii in xrange(maxlen):
            # Get next states
            # In the first iteration, we provide -1 and obtain the log_p's for the
            # first word. In the following iterations tiled_ctx becomes a batch
            # of duplicated left hypotheses. tiled_ctx is always the same except
            # the size of the 2nd dimension as the context vectors of the source
            # sequence is always the same regardless of the decoding process.
            # next_state's shape is (live_beam, rnn_dim)
            next_log_p, _, next_state, alphas = self.f_next(*[next_w, tiled_ctx, next_state])

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
            tiled_ctx   = np.tile(ctx0, [live_beam, 1])

        # dump every remaining hypotheses
        for idx in xrange(live_beam):
            final_sample.append(hyp_samples[idx])
            final_score.append(hyp_scores[idx])
            final_alignments.append(hyp_alignments[idx])

        return final_sample, final_score, final_alignments

    ###################################################################
    # The following methods can be redefined in child models inheriting
    # from this basic Attention model.
    ###################################################################
    def init_params(self):
        params = OrderedDict()

        # embedding weights for encoder and decoder
        params['Wemb_enc'] = norm_weight(self.n_words_src, self.embedding_dim, scale=self.weight_init)
        params['Wemb_dec'] = norm_weight(self.n_words_trg, self.embedding_dim, scale=self.weight_init)

        # encoder: bidirectional RNN
        #########
        # Forward encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder', nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init, layernorm=self.lnorm)
        # Backwards encoder
        params = get_new_layer(self.enc_type)[0](params, prefix='encoder_r', nin=self.embedding_dim, dim=self.rnn_dim, scale=self.weight_init, layernorm=self.lnorm)

        # How many additional encoder layers to add?
        for i in range(1, self.n_enc_layers):
            params = get_new_layer(self.enc_type)[0](params, prefix='deepencoder_%d' % i,
                                                     nin=self.ctx_dim, dim=self.ctx_dim,
                                                     scale=self.weight_init, layernorm=self.lnorm)

        # init_state, init_cell
        params = get_new_layer('ff')[0](params, prefix='ff_state', nin=self.ctx_dim, nout=self.rnn_dim, scale=self.weight_init)
        # decoder
        params = get_new_layer('gru_cond')[0](params, prefix='decoder', nin=self.embedding_dim, dim=self.rnn_dim, dimctx=self.ctx_dim, scale=self.weight_init, layernorm=False)

        # readout
        params = get_new_layer('ff')[0](params, prefix='ff_logit_gru'  , nin=self.rnn_dim       , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_prev' , nin=self.embedding_dim , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        params = get_new_layer('ff')[0](params, prefix='ff_logit_ctx'  , nin=self.ctx_dim       , nout=self.embedding_dim, scale=self.weight_init, ortho=False)
        if self.tied_trg_emb is False:
            params = get_new_layer('ff')[0](params, prefix='ff_logit'  , nin=self.embedding_dim , nout=self.n_words_trg, scale=self.weight_init)

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
        emb = dropout(self.tparams['Wemb_enc'][x.flatten()],
                      self.trng, self.emb_dropout, self.use_dropout)
        emb = emb.reshape([n_timesteps, n_samples, self.embedding_dim])
        proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder', mask=x_mask, layernorm=self.lnorm)

        # word embedding for backward rnn (source)
        embr = dropout(self.tparams['Wemb_enc'][xr.flatten()],
                       self.trng, self.emb_dropout, self.use_dropout)
        embr = embr.reshape([n_timesteps, n_samples, self.embedding_dim])
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r', mask=xr_mask, layernorm=self.lnorm)

        # context will be the concatenation of forward and backward rnns
        ctx = [tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)]

        for i in range(1, self.n_enc_layers):
            ctx = get_new_layer(self.enc_type)[1](self.tparams, ctx[0],
                                                  prefix='deepencoder_%d' % i,
                                                  mask=x_mask, layernorm=self.lnorm)



        # Apply dropout
        ctx = dropout(ctx[0], self.trng, self.ctx_dropout, self.use_dropout)

        # mean of the context (across time) will be used to initialize decoder rnn
        ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

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
        proj = get_new_layer('gru_cond')[1](self.tparams, emb,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state, layernorm=False)
        # hidden states of the decoder gru
        proj_h = proj[0]

        # weighted averages of context, generated by attention module
        ctxs = proj[1]

        # weights (alignment matrix)
        alphas = proj[2]

        # compute word probabilities
        logit_gru  = get_new_layer('ff')[1](self.tparams, proj_h, prefix='ff_logit_gru', activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, ctxs, prefix='ff_logit_ctx', activ='linear')
        logit_prev = get_new_layer('ff')[1](self.tparams, emb, prefix='ff_logit_prev', activ='linear')

        logit = dropout(tanh(logit_gru + logit_prev + logit_ctx), self.trng, self.out_dropout, self.use_dropout)

        if self.tied_trg_emb is False:
            logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        else:
            logit = tensor.dot(logit, self.tparams['Wemb_dec'].T)

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

        # For alpha regularization
        self.x_mask = x_mask
        self.y_mask = y_mask
        self.alphas = alphas

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
        proj = get_new_layer(self.enc_type)[1](self.tparams, emb, prefix='encoder', layernorm=self.lnorm)
        projr = get_new_layer(self.enc_type)[1](self.tparams, embr, prefix='encoder_r', layernorm=self.lnorm)

        # concatenate forward and backward rnn hidden states
        ctx = [tensor.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)]

        for i in range(1, self.n_enc_layers):
            ctx = get_new_layer(self.enc_type)[1](self.tparams, ctx[0],
                                                  prefix='deepencoder_%d' % i,
                                                  layernorm=self.lnorm)

        ctx      = ctx[0]
        # get the input for decoder rnn initializer mlp
        ctx_mean = ctx.mean(0)
        # ctx_mean = tensor.concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
        init_state = get_new_layer('ff')[1](self.tparams, ctx_mean, prefix='ff_state', activ='tanh')

        outs = [init_state, ctx]
        self.f_init = theano.function([x], outs, name='f_init')

        # x: 1 x 1
        y = tensor.vector('y_sampler', dtype=INT)
        init_state = tensor.matrix('init_state', dtype=FLOAT)

        # if it's the first word, emb should be all zero and it is indicated by -1
        emb = tensor.switch(y[:, None] < 0,
                            tensor.alloc(0., 1, self.tparams['Wemb_dec'].shape[1]),
                            self.tparams['Wemb_dec'][y])

        # apply one step of conditional gru with attention
        # get the next hidden states
        # get the weighted averages of contexts for this target word y
        r = get_new_layer('gru_cond')[1](self.tparams, emb,
                                         prefix='decoder',
                                         mask=None, context=ctx,
                                         one_step=True,
                                         init_state=init_state, layernorm=False)

        next_state = r[0]
        ctxs = r[1]
        alphas = r[2]

        logit_prev = get_new_layer('ff')[1](self.tparams, emb,          prefix='ff_logit_prev',activ='linear')
        logit_ctx  = get_new_layer('ff')[1](self.tparams, ctxs,         prefix='ff_logit_ctx', activ='linear')
        logit_gru  = get_new_layer('ff')[1](self.tparams, next_state,   prefix='ff_logit_gru', activ='linear')

        logit = tanh(logit_gru + logit_prev + logit_ctx)

        if self.tied_trg_emb is False:
            logit = get_new_layer('ff')[1](self.tparams, logit, prefix='ff_logit', activ='linear')
        else:
            logit = tensor.dot(logit, self.tparams['Wemb_dec'].T)

        # compute the logsoftmax
        next_log_probs = tensor.nnet.logsoftmax(logit)

        # Sample from the softmax distribution
        next_probs = tensor.exp(next_log_probs)
        next_word = self.trng.multinomial(pvals=next_probs).argmax(1)

        # compile a function to do the whole thing above
        # next hidden state to be used
        inputs = [y, ctx, init_state]
        outs = [next_log_probs, next_word, next_state, alphas]
        self.f_next = theano.function(inputs, outs, name='f_next')
