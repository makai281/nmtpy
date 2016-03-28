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

        # Collect options
        self.options = dict(self.__dict__)
        self.trg_idict = trg_idict

        self.set_nanguard()
        self.set_trng(seed)

        # Image feature dimension will be set after loading data
        self.img_dim = None

        # Not used for this model but a None necessary for nmt-translate
        self.n_timesteps = None

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
                                    do_mask=False)

        # Get image feature dimension
        self.img_dim = train_src_iterator.dim

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
                                    do_mask=False)

        # Create multi iterator
        self.valid_iterator = get_iterator("multi")([valid_src_iterator, valid_trg_iterator])

    def init_params(self):
        params = OrderedDict()

        # Target language embedding matrix
        params['Wemb'] = norm_weight(self.n_words_trg, self.trg_emb_dim)

        # Main LSTM block
        params = get_new_layer('lstm')[0](params, nin=self.trg_emb_dim, dim=self.rnn_dim, forget_bias=0, prefix='lstm_decoder')

        # FF layer adapting image feature space to word embedding dimension
        params = get_new_layer('ff')[0](params, prefix='ff_img2emb'    , nin=self.img_dim, nout=self.trg_emb_dim)
        params = get_new_layer('ff')[0](params, prefix='ff_lstm2softmax', nin=self.rnn_dim, nout=self.n_words_trg)

        self.initial_params = params

    def build(self):
        # Target sentences: n_timesteps, n_samples
        y = tensor.matrix('y', dtype=INT)
        y_flat = y.flatten()
        y_mask = tensor.matrix('y_mask', dtype='float32')

        # Volatile # of timesteps for target sentences
        n_timesteps_trg = y.shape[0]
        n_samples = y.shape[1]

        # image: n_samples, img_dim
        x_img = tensor.matrix('x_img', dtype='float32')

        # Store tensors
        self.inputs['x_img'] = x_img
        self.inputs['y'] = y
        self.inputs['y_mask'] = y_mask

        # Image context as initial input to the LSTM
        # x(-1): W_img2emb * CNN(Image) --> trg_emb_dim size vector
        x_m1 = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img2emb', activ='linear')

        # Fetch target word embeddings for the batch
        emb = self.tparams['Wemb'][y_flat]
        x = emb.reshape([n_timesteps_trg, n_samples, self.trg_emb_dim])

        # Make place for image context and <bos> indicator
        # x_shifted[0] <- img, x_shifted[1] <- <bos> all zeros
        # x_shifted has shape (n_timesteps_trg + 2 x n_samples x trg_emb_dim)
        x_shifted = tensor.alloc(0., x.shape[0] + 2, x.shape[1], x.shape[2])
        x_shifted = tensor.set_subtensor(x_shifted[2:], x)
        x_shifted = tensor.set_subtensor(x_shifted[0], x_m1)

        rval = get_new_layer('lstm')[1](self.tparams, x_shifted, prefix='lstm_decoder')
        # lstm returns memory state m(t) and cell state c(t)
        # for each sequence in the batch: (n_trg_timesteps, n_samples, rnn_dim)
        m_t = rval[0]

        # This prepares m(t) for softmax
        logit = get_new_layer('ff')[1](self.tparams, m_t, prefix='ff_lstm2softmax', activ='linear')

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

        # Apply logsoftmax (stable version)
        next_log_probs = tensor.nnet.logsoftmax(
                logit.reshape([logit.shape[0]*logit.shape[1], logit.shape[2]]))

        # cost
        y_flat_idx = tensor.arange(y_flat.shape[0]) * self.n_words_trg + y_flat

        cost = -next_log_probs.flatten()[y_flat_idx]
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
        x_img  = tensor.matrix('x_img', dtype='float32')
        y_prev = tensor.vector('y_sampler', dtype=INT)

        # Image context as initial input to the LSTM
        # x(-1): W_img2emb * CNN(Image) --> trg_emb_dim size vector
        x_m1 = get_new_layer('ff')[1](self.tparams, x_img, prefix='ff_img2emb', activ='linear')

        # Pass x_m1 through lstm to obtain the initial memory and state
        rval = get_new_layer('lstm')[1](self.tparams, x_m1, one_step=True, prefix='lstm_decoder')
        m_0, c_0 = rval

        # initial state
        self.f_init = theano.function([x_img], [m_0, c_0], name='f_init', profile=self.profile)

        # if it's the first word, emb should be all zero and it is indicated by -1
        emb = tensor.switch(y_prev[:, None] < 0,
                            tensor.alloc(0., 1, self.trg_emb_dim),
                            self.tparams['Wemb'][y_prev])

        # apply one step of LSTM
        # init_memory will be an input to f_next() by the caller
        init_memory = tensor.matrix('init_memory', dtype='float32')
        init_state  = tensor.matrix('init_state' , dtype='float32')
        rval = get_new_layer('lstm')[1](self.tparams, emb, one_step=True,
                                        init_memory=init_memory, init_state=init_state,
                                        prefix='lstm_decoder')

        m_t, c_t = rval

        # This prepares m(t) for softmax
        logit = get_new_layer('ff')[1](self.tparams, m_t, prefix='ff_lstm2softmax', activ='linear')

        if self.dropout > 0:
            logit = dropout_layer(logit, self.use_dropout, self.dropout, self.trng)

        # Apply logsoftmax (stable version)
        next_log_probs = tensor.nnet.logsoftmax(logit)

        # init_memory and init_state has dims 1 x rnn_dim for the first call
        inputs  = [y_prev, init_memory, init_state]
        outs    = [next_log_probs, m_t, c_t]

        self.f_next = theano.function(inputs, outs, name='f_next', profile=self.profile)

    def beam_search(self, inputs, beam_size=12, maxlen=50):
        # Final results and their scores
        final_sample = []
        final_score  = []

        live_beam = 1
        dead_beam = 0

        # Initially we have one empty hypothesis
        # with a score of 0
        hyp_states  = []
        hyp_samples = [[]]
        hyp_memories= []
        hyp_scores  = np.array([0.], dtype=FLOAT)

        # Give the image context as initial input and
        # receive the linearly transformed image embedding
        # next_state (m(t)) is x(-1), e.g. initial image context linearly
        # transformed to match embedding size
        next_memory, next_state = self.f_init(inputs[0])

        # Beginning-of-sentence indicator is -1
        # This is detected by build_sampler() to put all 0's
        # for the first word of the sentence
        next_w = np.array([-1.], dtype=INT)

        for ii in xrange(maxlen):
            # Get next states
            inputs = [next_w, next_memory, next_state]
            next_log_p, next_memory, next_state = self.f_next(*inputs)

            # Beam search
            cand_scores = hyp_scores[:, None] - next_log_p
            cand_flat = cand_scores.flatten()

            # Take the best beam_size-dead_beam hypotheses
            ranks_flat = cand_flat.argsort()[:(beam_size-dead_beam)]
            # Get their costs
            costs = cand_flat[ranks_flat]

            voc_size = next_log_p.shape[1]
            # Find out to which hypothesis idx this was belonging
            trans_indices = ranks_flat / voc_size
            # Find out the just added word idx
            word_indices = ranks_flat % voc_size

            # New states, scores and samples
            new_hyp_states  = []
            new_hyp_samples = []
            new_hyp_memories= []
            new_hyp_scores  = np.zeros(beam_size-dead_beam).astype('float32')

            # Iterate over the hypotheses
            # and add them to new_* lists
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                new_hyp_memories.append(copy.copy(next_memory[ti]))

            # check the finished samples
            new_live_beam = 0
            hyp_memories = []
            hyp_samples  = []
            hyp_scores   = []
            hyp_states   = []

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
                    hyp_memories.append(new_hyp_memories[idx])

            hyp_scores = np.array(hyp_scores)
            live_beam = new_live_beam

            if new_live_beam < 1:
                break
            if dead_beam >= beam_size:
                break

            # Prepare for the next iteration
            next_w = np.array([w[-1] for w in hyp_samples])
            next_state = np.array(hyp_states)
            next_memory= np.array(hyp_memories)

        # dump every remaining hypotheses
        if live_beam > 0:
            for idx in xrange(live_beam):
                final_sample.append(hyp_samples[idx])
                final_score.append(hyp_scores[idx])

        return final_sample, final_score
