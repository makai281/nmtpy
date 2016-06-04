import numpy as np

import theano
from theano import tensor

from .nmtutils import *
from .nmtutils import _p
from .typedef import *

# Shorthands for activations
linear  = lambda x: x
sigmoid = tensor.nnet.sigmoid
tanh    = tensor.tanh
relu    = tensor.nnet.relu

##################
# GRU layer step()
##################
# sequences:
#   m_    : mask
#   x_    : state_below_
#   xx_   : state_belowx
# outputs-info:
#   h_    : init_states
# non-seqs:
#   U     : shared U matrix
#   Ux    : shared Ux matrix
def gru_step(m_, x_, xx_, h_, U, Ux):
    # sigmoid([U_r * h_ + (W_r * X + b_r) , U_z * h_ + (W_z * X + b_z)])
    preact = sigmoid(tensor.dot(h_, U) + x_)

    # slice reset and update gates
    r = _tensor_slice(preact, 0, Ux.shape[1])
    u = _tensor_slice(preact, 1, Ux.shape[1])

    # NOTE: Is this correct or should be tensor.dot(h_ * r, Ux) ?
    # hidden state proposal (h_tilda_j eq. 8)
    h_tilda = tanh(((tensor.dot(h_, Ux)) * r) + xx_)

    # leaky integrate and obtain next hidden state
    # According to paper, this should be [h = u * h_tilda + (1 - u) * h_]
    h = u * h_tilda + (1. - u) * h_
    # What's the logic to invert the mask and add it after multiplying by h_?
    # -> h is new h if mask is not 0 (a word was presented), otherwise, h is the copy of previous h which is h_
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h

def _tensor_slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    elif _x.ndim == 2:
        return _x[:, n*dim:(n+1)*dim]
    return _x[n*dim:(n+1)*dim]

#########
# dropout
#########
def dropout_layer(state_before, use_dropout, dropout_prob, trng):
    proj = tensor.switch(
        use_dropout,
        state_before * trng.binomial(state_before.shape, p=dropout_prob, n=1,
                                     dtype=state_before.dtype),
        state_before * dropout_prob)
    return proj

###############################################
# Returns the initializer and the layer itself
###############################################
def get_new_layer(name):
    # Layer type: (initializer, layer)
    layers = {
                'ff'                : ('param_init_fflayer'     , 'fflayer'),
                'gru'               : ('param_init_gru'         , 'gru_layer'),
                'gru_cond'          : ('param_init_gru_cond'    , 'gru_cond_layer'),
                'gru_cond_multi'    : ('param_init_gru_cond'    , 'gru_cond_multi_layer'),
                'lstm'              : ('param_init_lstm'        , 'lstm_layer'),
                'lstm_cond'         : ('param_init_lstm_cond'   , 'lstm_cond_layer'),
             }

    init, layer = layers[name]
    return (eval(init), eval(layer))

#####################################################################
# feedforward layer: affine transformation + point-wise nonlinearity
#####################################################################
def param_init_fflayer(params, nin, nout, scale=0.01, ortho=True, prefix='ff'):
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=scale, ortho=ortho)
    params[_p(prefix, 'b')] = np.zeros((nout,)).astype(FLOAT)

    return params

def fflayer(tparams, state_below, prefix='ff', activ='tanh'):
    return eval(activ) (
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')]
        )

###########
# GRU layer
###########
def param_init_gru(params, nin, dim, scale=0.01, prefix='gru'):
    # embedding to gates transformation weights, biases

    # See the paper for variable names
    # W is stacked W_r and W_z
    params[_p(prefix, 'W')]  = np.concatenate([norm_weight(nin, dim, scale=scale),
                                               norm_weight(nin, dim, scale=scale)], axis=1)
    # b_r and b_z
    params[_p(prefix, 'b')]  = np.zeros((2 * dim,)).astype(FLOAT)

    # recurrent transformation weights for gates
    # U is stacked U_r and U_z
    params[_p(prefix, 'U')]  = np.concatenate([ortho_weight(dim),
                                               ortho_weight(dim)], axis=1)

    # embedding to hidden state proposal weights, biases
    # The followings appears in eq 8 where we compute the candidate h (tilde)
    params[_p(prefix, 'Wx')] = norm_weight(nin, dim, scale=scale)
    params[_p(prefix, 'bx')] = np.zeros((dim,)).astype(FLOAT)

    # recurrent transformation weights for hidden state proposal
    params[_p(prefix, 'Ux')] = ortho_weight(dim)

    return params


def gru_layer(tparams, state_below, prefix='gru', mask=None, profile=False, mode=None):
    nsteps = state_below.shape[0]

    # if we are dealing with a mini-batch
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1

    # Infer RNN dimensionality
    dim = tparams[_p(prefix, 'Ux')].shape[1]

    # if we have no mask, we assume all the inputs are valid
    if mask is None:
        # tensor.alloc(value, *shape)
        # mask: (n_steps, 1) filled with 1
        mask = tensor.alloc(1., nsteps, 1)

    # state_below is the input word embeddings
    # input to the gates, concatenated
    # [W_r * X + b_r, W_z * X + b_z]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    # input to compute the hidden state proposal
    # This is the [W*x]_j in the eq. 8 of the paper
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(gru_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                mode=mode,
                                strict=True)
    rval = [rval]
    return rval

######################################
# Conditional GRU layer with Attention
######################################
def param_init_gru_cond(params, nin, dim, dimctx, scale=0.01, prefix='gru_cond',
                        nin_nonlin=None, dim_nonlin=None):
    # nin:      input dim (e.g. embedding dim in the case of NMT)
    # dim:      gru_dim   (e.g. 1000)
    # dimctx:   2*gru_dim (e.g. 2000)

    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    # Below ones area also available in gru_layer
    params[_p(prefix, 'W')]             = np.concatenate([norm_weight(nin, dim, scale=scale),
                                                          norm_weight(nin, dim, scale=scale)], axis=1)
    params[_p(prefix, 'b')]             = np.zeros((2 * dim,)).astype(FLOAT)

    params[_p(prefix, 'U')]             = np.concatenate([ortho_weight(dim_nonlin),
                                                          ortho_weight(dim_nonlin)], axis=1)

    params[_p(prefix, 'Wx')]            = norm_weight(nin_nonlin, dim_nonlin, scale=scale)
    params[_p(prefix, 'Ux')]            = ortho_weight(dim_nonlin)
    params[_p(prefix, 'bx')]            = np.zeros((dim_nonlin,)).astype(FLOAT)

    # Below ones are new to this layer
    params[_p(prefix, 'U_nl')]          = np.concatenate([ortho_weight(dim_nonlin),
                                                          ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'b_nl')]          = np.zeros((2 * dim_nonlin,)).astype(FLOAT)

    params[_p(prefix, 'Ux_nl')]         = ortho_weight(dim_nonlin)
    params[_p(prefix, 'bx_nl')]         = np.zeros((dim_nonlin,)).astype(FLOAT)

    # context to GRU
    params[_p(prefix, 'Wc')]            = norm_weight(dimctx, dim*2, scale=scale)
    params[_p(prefix, 'Wcx')]           = norm_weight(dimctx, dim, scale=scale)

    # attention: combined -> hidden
    params[_p(prefix, 'W_comb_att')]    = norm_weight(dim, dimctx, scale=scale)

    # attention: context -> hidden
    # attention: hidden bias
    params[_p(prefix, 'Wc_att')]        = norm_weight(dimctx, dimctx, scale=scale)
    params[_p(prefix, 'b_att')]         = np.zeros((dimctx,)).astype(FLOAT)

    # attention: This gives the alpha's
    params[_p(prefix, 'U_att')]         = norm_weight(dimctx, 1, scale=scale)
    params[_p(prefix, 'c_att')]         = np.zeros((1,)).astype(FLOAT)

    return params

def gru_cond_layer(tparams, state_below, context, prefix='gru_cond',
                   mask=None, one_step=False,
                   init_state=None, context_mask=None,
                   profile=False, mode=None):
    if one_step:
        assert init_state, 'previous state must be provided'

    # Context
    # n_timesteps x n_samples x ctxdim
    assert context, 'Context must be provided'
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'

    nsteps = state_below.shape[0]

    # Batch or single sample?
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1

    # if we have no mask, we assume all the inputs are valid
    # tensor.alloc(value, *shape)
    # mask: (n_steps, 1) filled with 1
    if mask is None:
        mask = tensor.alloc(1., nsteps, 1)

    # Infer RNN dimensionality
    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    # if not given, assume it's all zeros
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # These two dot products are same with gru_layer, refer to the equations.
    # [W_r * X + b_r, W_z * X + b_z]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    # input to compute the hidden state proposal
    # This is the [W*x]_j in the eq. 8 of the paper
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    # Wc_att: dimctx -> dimctx
    # Linearly transform the context to another space with same dimensionality
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + tparams[_p(prefix, 'b_att')]

    # Step function for the recurrence/scan
    # Sequences
    # ---------
    # m_    : mask
    # x_    : state_below_
    # xx_   : state_belowx
    # outputs_info
    # ------------
    # h_    : init_state,
    # ctx_  : need to be defined as it's returned by _step
    # alpha_: need to be defined as it's returned by _step
    # non sequences
    # -------------
    # pctx_ : pctx_
    # cc_   : context
    # and all the shared weights and biases..

    def _step(m_, x_, xx_,
              h_, ctx_, alpha_,
              pctx_, cc_, U, Wc, W_comb_att, U_att, c_att, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):

        # Do a step of classical GRU
        h1 = gru_step(m_, x_, xx_, h_, U, Ux)

        ###########
        # Attention
        ###########
        # h1 X W_comb_att
        # W_comb_att: dim -> dimctx
        # pstate_ should be 2D as we're working with unrolled timesteps
        pstate_ = tensor.dot(h1, W_comb_att)

        # Accumulate in pctx__ and apply tanh()
        # This becomes the projected context + the current hidden state
        # of the decoder, e.g. this is the information accumulating
        # into the returned original contexts with the knowledge of target
        # sentence decoding.
        pctx__ = tanh(pctx_ + pstate_[None, :, :])

        # Affine transformation for alpha = (pctx__ X U_att) + c_att
        # We're now down to scalar alpha's for each accumulated
        # context (0th dim) in the pctx__
        # alpha should be n_timesteps, 1, 1
        alpha = tensor.dot(pctx__, U_att) + c_att

        # Drop the last dimension, e.g. (n_timesteps, 1)
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])

        # Exponentiate alpha
        alpha = tensor.exp(alpha)

        # If there is a context mask, multiply with it to cancel unnecessary steps
        if context_mask:
            alpha = alpha * context_mask

        # Normalize so that the sum makes 1
        alpha = alpha / (alpha.sum(0, keepdims=True) + 1e-6)

        # Compute the current context ctx_ as the alpha-weighted sum of
        # the initial contexts: context
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        ###########################################
        # ctx_ and alpha computations are completed
        ###########################################

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
        r2 = _tensor_slice(preact, 0, dim)
        u2 = _tensor_slice(preact, 1, dim)

        preactx = (tensor.dot(h1, Ux_nl) + bx_nl) * r2
        preactx += tensor.dot(ctx_, Wcx)

        # Candidate hidden
        h2_tilda = tanh(preactx)

        # Leaky integration between the new h2 and the
        # old h1 computed in line 285
        h2 = u2 * h2_tilda + (1. - u2) * h1
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx]

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_att')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] + shared_vars))
    else:
        outputs_info=[init_state,
                      tensor.alloc(0., n_samples, context.shape[2]), # ctxdim       (ctx_)
                      tensor.alloc(0., n_samples, context.shape[0])] # n_timesteps  (alpha)

        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=[pctx_, context] + shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    mode=mode,
                                    strict=True)
    return rval

###########################################################
# Conditional GRU layer with multiple context and attention
###########################################################
def gru_cond_multi_layer(tparams, state_below, ctx1, ctx2, prefix='gru_cond_multi',
                         input_mask=None, one_step=False,
                         init_state=None, ctx1_mask=None,
                         profile=False, mode=None):
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
    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    # if not given, assume it's all zeros
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # These two dot products are same with gru_layer, refer to the equations.
    # [W_r * X + b_r, W_z * X + b_z]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    # input to compute the hidden state proposal
    # This is the [W*x]_j in the eq. 8 of the paper
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    # Wc_att: dimctx -> dimctx
    # Linearly transform the contexts to another space with same dimensionality
    pctx1_ = tensor.dot(ctx1, tparams[_p(prefix, 'Wc_att')]) + tparams[_p(prefix, 'b_att')]
    pctx2_ = tensor.dot(ctx2, tparams[_p(prefix, 'Wc_att')]) + tparams[_p(prefix, 'b_att')]

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
              pctx1_, pctx2_, cc1_, cc2_, U, Wc, W_comb_att, U_att, c_att, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):

        # Do a step of classical GRU
        h1 = gru_step(m_, x_, xx_, h_, U, Ux)

        ###########
        # Attention
        ###########
        # h1 X W_comb_att
        # W_comb_att: dim -> dimctx
        # pstate_ should be 2D as we're working with unrolled timesteps
        pstate_ = tensor.dot(h1, W_comb_att)

        # Accumulate in pctx*__ and apply tanh()
        # This becomes the projected context(s) + the current hidden state
        # of the decoder, e.g. this is the information accumulating
        # into the returned original contexts with the knowledge of target
        # sentence decoding.
        pctx1__ = tanh(pctx1_ + pstate_[None, :, :])
        pctx2__ = tanh(pctx2_ + pstate_[None, :, :])

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
        alpha1 = tensor.exp(alpha1)
        alpha2 = tensor.exp(alpha2)

        # If there is a context mask, multiply with it to cancel unnecessary steps
        # We won't have a ctx_mask for image vectors
        if ctx1_mask:
            alpha1 = alpha1 * ctx1_mask

        # Normalize so that the sum makes 1
        alpha1 = alpha1 / (alpha1.sum(0, keepdims=True) + 1e-6)
        alpha2 = alpha2 / (alpha2.sum(0, keepdims=True) + 1e-6)

        # Compute the current context ctx*_ as the alpha-weighted sum of
        # the initial contexts ctx*'s
        ctx1_ = (cc1_ * alpha1[:, :, None]).sum(0)
        ctx2_ = (cc2_ * alpha2[:, :, None]).sum(0)

        # Sum the weighted-contexts and apply tanh()
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
        r2 = _tensor_slice(preact, 0, dim)
        u2 = _tensor_slice(preact, 1, dim)

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
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_att')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

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
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    mode=mode,
                                    strict=True)
    return rval


#################
# LSTM (from SAT)
#################
def param_init_lstm(params, nin, dim, forget_bias=0, scale=0.01, prefix='lstm'):
    """
     Stack the weight matrices for all the gates
     for much cleaner code and slightly faster dot-prods
    """
    # input weights
    # W_ix: Input x to input gate
    # W_fx: Input x to forget gate
    # W_ox: Input x to output gate
    # W_cx: Input x to cell state
    params[_p(prefix, 'W')] = np.concatenate([norm_weight(nin, dim, scale=scale),
                                              norm_weight(nin, dim, scale=scale),
                                              norm_weight(nin, dim, scale=scale),
                                              norm_weight(nin, dim, scale=scale)], axis=1)

    # for the previous hidden activation
    # W_im: Memory t_1 to input(t)
    # W_fm: Memory t_1 to forget(t)
    # W_om: Memory t_1 to output(t)
    # W_cm: Memory t_1 to cellstate(t)
    params[_p(prefix, 'U')] = np.concatenate([ortho_weight(dim),
                                              ortho_weight(dim),
                                              ortho_weight(dim),
                                              ortho_weight(dim)], axis=1)

    b = np.zeros((4 * dim,)).astype(FLOAT)
    b[dim: 2*dim] = forget_bias
    params[_p(prefix, 'b')] = b

    return params

# This function implements the lstm fprop
def lstm_layer(tparams, state_below, init_state=None, init_memory=None, one_step=False, prefix='lstm'):

    #if one_step:
    #    assert init_memory, 'previous memory must be provided'
    #    assert init_state, 'previous state must be provided'

    # number of timesteps
    nsteps = state_below.shape[0]

    # hidden dimension of LSTM layer
    dim = tparams[_p(prefix, 'U')].shape[0]

    if state_below.ndim == 3:
        # This is minibatch
        n_samples = state_below.shape[1]
    else:
        # during sampling, only single sample is received
        n_samples = 1

    if init_state is None:
        # init_state is dim per sample all zero
        init_state = tensor.alloc(0., n_samples, dim)

    if init_memory is None:
        # init_memory is dim per sample all zero
        init_memory = tensor.alloc(0., n_samples, dim)

    # This maps the input to LSTM dimensionality
    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    ###########################
    # one time step of the lstm
    ###########################
    def _step(x_, m_, c_):
        """
           x_: state_below
           m_: init_memory
           c_: init_cell_state (this is actually not used when initializing)
        """
        
        preact = tensor.dot(m_, tparams[_p(prefix, 'U')])
        preact += x_

        # input(t) = sigm(W_ix * x_t + W_im * m_tm1)
        i = sigmoid(_tensor_slice(preact, 0, dim))
        f = sigmoid(_tensor_slice(preact, 1, dim))
        o = sigmoid(_tensor_slice(preact, 2, dim))

        # cellstate(t)?
        c = tanh(_tensor_slice(preact, 3, dim))

        # cellstate(t) = forget(t) * cellstate(t-1) + input(t) * cellstate(t)
        c = f * c_ + i * c

        # m_t, e.g. memory in tstep T in NIC paper
        m = o * tanh(c)

        return m, c

    if one_step:
        rval = _step(state_below, init_memory, init_state)
    else:
        rval, updates = theano.scan(_step,
                                    sequences=[state_below],
                                    outputs_info=[init_memory, init_state],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps, profile=False)
    return rval

#######################################
# Conditional LSTM layer with Attention
#######################################
def param_init_lstm_cond(params, options, nin, dim, dimctx, scale=0.01, prefix='lstm_cond'):
    # input to LSTM, similar to the above, we stack the matrices for compactness, do one
    # dot product, and use the slice function below to get the activations for each "gate"
    params[_p(prefix,'W')] = np.concatenate([norm_weight(nin, dim, scale=scale),
                                             norm_weight(nin, dim, scale=scale),
                                             norm_weight(nin, dim, scale=scale),
                                             norm_weight(nin, dim, scale=scale)], axis=1)

    # LSTM to LSTM
    params[_p(prefix,'U')] = np.concatenate([ortho_weight(dim),
                                             ortho_weight(dim),
                                             ortho_weight(dim),
                                             ortho_weight(dim)], axis=1)

    # bias to LSTM
    params[_p(prefix,'b')] = np.zeros((4 * dim,)).astype(FLOAT)

    # context to LSTM
    params[_p(prefix,'Wc')] = norm_weight(dimctx, dim*4, scale=scale)

    # attention: context -> hidden
    params[_p(prefix,'Wc_att')] = norm_weight(dimctx, dimctx, scale=scale, ortho=False)

    # attention: LSTM -> hidden
    params[_p(prefix,'Wd_att')] = norm_weight(dim, dimctx, scale=scale)

    # attention: hidden bias
    params[_p(prefix,'b_att')] = np.zeros((dimctx,)).astype(FLOAT)

    # optional "deep" attention
    if options['n_layers_att'] > 1:
        for lidx in xrange(1, options['n_layers_att']):
            params[_p(prefix, 'W_att_%d' % lidx)] = ortho_weight(dimctx)
            params[_p(prefix, 'b_att_%d' % lidx)] = np.zeros((dimctx,)).astype(FLOAT)

    # attention:
    params[_p(prefix,'U_att')] = norm_weight(dimctx, 1, scale=scale)
    params[_p(prefix, 'c_tt')] = np.zeros((1,)).astype(FLOAT)

    if options['selector']:
        # attention: selector
        params[_p(prefix, 'W_sel')] = norm_weight(dim, 1, scale=scale)
        params[_p(prefix, 'b_sel')] = np.float32(0.)

    return params

def lstm_cond_layer(tparams, state_below, options, prefix='lstm',
                    mask=None, context=None, one_step=False,
                    init_memory=None, init_state=None,
                    trng=None, use_noise=None):

    assert context, 'Context must be provided'

    if one_step:
        assert init_memory, 'previous memory must be provided'
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # infer lstm dimension
    dim = tparams[_p(prefix, 'U')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # projected context
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix, 'b_att')]

    # Multiple LSTM layers in attention?
    if options['n_layers_att'] > 1:
        for lidx in xrange(1, options['n_layers_att']):
            pctx_ = tensor.dot(pctx_, tparams[_p(prefix,'W_att_%d'%lidx)])+tparams[_p(prefix, 'b_att_%d'%lidx)]
            # note to self: this used to be options['n_layers_att'] - 1, so no extra non-linearity if n_layers_att < 3
            if lidx < options['n_layers_att']:
                pctx_ = tanh(pctx_)

    # projected x
    # state_below is timesteps*num samples by d in training (TODO change to notation of paper)
    # this is n * d during sampling
    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    def _step(m_, x_, h_, c_, ct_, pctx_):
        """ Each variable is one time slice of the LSTM
        m_ - (mask), x_- (previous word), h_- (hidden state), c_- (lstm memory), ct_- (context),
        pctx_ (projected context)
        """
        # attention computation
        # [described in  equations (4), (5), (6) in
        # section "3.1.2 Decoder: Long Short Term Memory Network]
        pstate_ = tensor.dot(h_, tparams[_p(prefix,'Wd_att')])
        pctx_ = pctx_ + pstate_[:, None, :]
        pctx_list = []
        pctx_list.append(pctx_)
        pctx_ = tanh(pctx_)
        alpha = tensor.dot(pctx_, tparams[_p(prefix,'U_att')]) + tparams[_p(prefix, 'c_tt')]
        alpha_pre = alpha
        alpha_shp = alpha.shape

        # Soft attention
        alpha = tensor.nnet.softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
        ctx_ = (context * alpha[:,:,None]).sum(1) # current context
        alpha_sample = alpha # you can return something else reasonable here to debug

        if options['selector']:
            sel_ = sigmoid(tensor.dot(h_, tparams[_p(prefix, 'W_sel')])+tparams[_p(prefix,'b_sel')])
            sel_ = sel_.reshape([sel_.shape[0]])
            ctx_ = sel_[:,None] * ctx_

        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tensor.dot(ctx_, tparams[_p(prefix, 'Wc')])

        # Recover the activations to the lstm gates
        # [equation (1)]
        i = _tensor_slice(preact, 0, dim)
        f = _tensor_slice(preact, 1, dim)
        o = _tensor_slice(preact, 2, dim)
        i = sigmoid(i)
        f = sigmoid(f)
        o = sigmoid(o)
        c = tanh(_tensor_slice(preact, 3, dim))

        # compute the new memory/hidden state
        # if the mask is 0, just copy the previous state
        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        rval = [h, c, alpha, alpha_sample, ctx_]
        if options['selector']:
            rval += [sel_]
        rval += [pstate_, pctx_, i, f, o, preact, alpha_pre] + pctx_list
        return rval

    if options['selector']:
        _step0 = lambda m_, x_, h_, c_, ct_, sel_, pctx_: _step(m_, x_, h_, c_, ct_, pctx_)
    else:
        _step0 = lambda m_, x_, h_, c_, ct_, pctx_: _step(m_, x_, h_, c_, ct_, pctx_)

    if one_step:
        if options['selector']:
            rval = _step0(mask, state_below, init_state, init_memory, None, None, pctx_)
        else:
            rval = _step0(mask, state_below, init_state, init_memory, None, pctx_)
        return rval
    else:
        seqs = [mask, state_below]
        if options['selector']:
            outputs_info += [tensor.alloc(0., n_samples)]
        outputs_info += [None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None] + [None] # *options['n_layers_att']
        rval, updates = theano.scan(_step0,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=[pctx_],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps, profile=False)
        return rval, updates
