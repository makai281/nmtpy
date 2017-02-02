# -*- coding: utf-8 -*-
import numpy as np

import theano
import theano.tensor as tensor

# Ours
from ..layers import *
from ..defaults import INT, FLOAT

# Base fusion model
from .basefusion import Model as Fusion

class Model(Fusion):
    def __init__(self, seed, logger, **kwargs):
        # Call parent's init first
        super(Model, self).__init__(seed, logger, **kwargs)

        # Set architecture specific methods
        self.init_gru_decoder   = init_gru_decoder_multi
        self.gru_decoder        = gru_decoder_multi

########## Define layers here ###########
def init_gru_decoder_multi(params, nin, dim, dimctx, scale=0.01, prefix='gru_decoder_multi'):
    # Init with usual gru_cond function
    params = param_init_gru_cond(params, nin, dim, dimctx, scale, prefix)

    # Add separate attention weights for the 2nd modality
    params[pp(prefix, 'Wc_att2')] = norm_weight(dimctx, dimctx, scale=scale)
    params[pp(prefix,  'b_att2')] = np.zeros((dimctx,)).astype(FLOAT)

    # attention: This gives the alpha's
    params[pp(prefix, 'U_att2')] = norm_weight(dimctx, 1, scale=scale)
    params[pp(prefix, 'c_att2')] = np.zeros((1,)).astype(FLOAT)
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
              pctx1_, pctx2_, cc1_, cc2_, U, Wc, W_comb_att, W_comb_att2,
              U_att, c_att, U_att2, c_att2,
              Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):

        # Do a step of classical GRU
        h1 = gru_step(m_, x_, xx_, h_, U, Ux)

        ###########
        # Attention
        ###########
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
        alpha2 = tensor.dot(pctx2__, U_att2) + c_att2

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

        # Sum of contexts
        ctx_ = tanh(ctx1_ + ctx2_)

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
                   tparams[pp(prefix, 'U_att2')],
                   tparams[pp(prefix, 'c_att2')],
                   tparams[pp(prefix, 'Ux')],
                   tparams[pp(prefix, 'Wcx')],
                   tparams[pp(prefix, 'U_nl')],
                   tparams[pp(prefix, 'Ux_nl')],
                   tparams[pp(prefix, 'b_nl')],
                   tparams[pp(prefix, 'bx_nl')]]

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


