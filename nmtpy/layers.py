import numpy as np

import theano
from theano import tensor

from .nmtutils import *
from .nmtutils import _p

# Shorthands for activations
tanh = tensor.tanh
relu = tensor.nnet.relu
linear = lambda x: x

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
                'ff'        : ('param_init_fflayer', 'fflayer'),
                'gru'       : ('param_init_gru', 'gru_layer'),
                'gru_cond'  : ('param_init_gru_cond', 'gru_cond_layer'),
             }

    init, layer = layers[name]
    return (eval(init), eval(layer))

#####################################################################
# feedforward layer: affine transformation + point-wise nonlinearity
#####################################################################
def param_init_fflayer(params, prefix='ff', nin=None, nout=None, ortho=True):
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, prefix='ff',
            activ='lambda x: tanh(x)', **kwargs):
    return eval(activ) (
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')]
        )

###########
# GRU layer
###########
def param_init_gru(params, prefix='gru', nin=None, dim=None):
    # embedding to gates transformation weights, biases
    params[_p(prefix, 'W')]  = np.concatenate([norm_weight(nin, dim),
                                               norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'b')]  = np.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    params[_p(prefix, 'U')]  = np.concatenate([ortho_weight(dim),
                                               ortho_weight(dim)], axis=1)

    # embedding to hidden state proposal weights, biases
    params[_p(prefix, 'Wx')] = norm_weight(nin, dim)
    params[_p(prefix, 'bx')] = np.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    params[_p(prefix, 'Ux')] = ortho_weight(dim)

    return params


def gru_layer(tparams, state_below, prefix='gru', mask=None, profile=False, mode=None, **kwargs):
    nsteps = state_below.shape[0]

    # if we are dealing with a mini-batch
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    # during sampling
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    # if we have no mask, we assume all the inputs are valid
    if mask is None:
        # tensor.alloc(value, *shape)
        # mask: (n_steps, 1) filled with 1
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    ##############################################################
    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # step function to be used by scan
    # sequences:
    #   m_    : mask
    #   x_    : state_below_
    #   xx_   : state_belowx
    # outputs-info:
    #   h_    : init_states
    # non-seqs:
    #   U     : shared U matrix
    #   Ux    : shared Ux matrix
    def _step(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h
    ##############################################################

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
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


# Conditional GRU layer with Attention
def param_init_gru_cond(params, nin, dim, dimctx, prefix='gru_cond',
                        nin_nonlin=None, dim_nonlin=None):

    # By default dimctx is 2*gru_dim, e.g. 2000
    # nin: embedding_dim
    # dim: gru_dim

    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    params[_p(prefix, 'W')]             = np.concatenate([norm_weight(nin, dim),
                                                          norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'b')]             = np.zeros((2 * dim,)).astype('float32')

    params[_p(prefix, 'U')]             = np.concatenate([ortho_weight(dim_nonlin),
                                                          ortho_weight(dim_nonlin)], axis=1)

    params[_p(prefix, 'Wx')]            = norm_weight(nin_nonlin, dim_nonlin)

    params[_p(prefix, 'Ux')]            = ortho_weight(dim_nonlin)
    params[_p(prefix, 'bx')]            = np.zeros((dim_nonlin,)).astype('float32')

    params[_p(prefix, 'U_nl')]          = np.concatenate([ortho_weight(dim_nonlin),
                                                          ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'b_nl')]          = np.zeros((2 * dim_nonlin,)).astype('float32')

    params[_p(prefix, 'Ux_nl')]         = ortho_weight(dim_nonlin)
    params[_p(prefix, 'bx_nl')]         = np.zeros((dim_nonlin,)).astype('float32')

    # context to GRU
    params[_p(prefix, 'Wc')]            = norm_weight(dimctx, dim*2)

    params[_p(prefix, 'Wcx')]           = norm_weight(dimctx, dim)

    # attention: combined -> hidden
    params[_p(prefix, 'W_comb_att')]    = norm_weight(dim, dimctx)

    # attention: context -> hidden
    params[_p(prefix, 'Wc_att')]        = norm_weight(dimctx)

    # attention: hidden bias
    params[_p(prefix, 'b_att')]         = np.zeros((dimctx,)).astype('float32')

    # attention: This gives the alpha's
    params[_p(prefix, 'U_att')]         = norm_weight(dimctx, 1)
    params[_p(prefix, 'c_att')]         = np.zeros((1,)).astype('float32')

    return params


def gru_cond_layer(tparams, state_below, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None, profile=False, mode=None,
                   **kwargs):

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]
    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    # projected context
    # n_timesteps x n_samples x ctxdim
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        # We have a batch
        n_samples = state_below.shape[1]
    else:
        # Single sample
        n_samples = 1

    # if we have no mask, we assume all the inputs are valid
    # tensor.alloc(value, *shape)
    # mask: (n_steps, 1) filled with 1
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    # if not given, assume it's all zeros
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # A dot operation between 3D context and 2D weights Wc_att
    # results in application of the non-linearity with
    # the last dimension of the context (ctx_dim)
    # Final shape remains the same pctx_.shape == context.shape
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + tparams[_p(prefix, 'b_att')]

    # projected x
    # state_below is the target embeddings. Here I think two
    # non-linearities are applied to the same state_below.
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    # Step function for the recurrence/scan
    # m_: mask, x_: state_below_, xx_: state_belowx (sequences)
    # h_: init_state, ctx_: 0 or None, alpha_: 0 or None (outputs_info)
    # pctx_: pctx_, cc_: context (non_sequences)
    def _step(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
              # The followings are the shared variables
              U, Wc, W_comb_att, U_att, c_att, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):

        # init_state X U + state_below_
        preact1 = tensor.dot(h_, U) + x_
        preact1 = tensor.nnet.sigmoid(preact1)

        # Slice activations
        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        # (init_state X Ux) * r1 + state_belowx
        preactx1 = (tensor.dot(h_, Ux) * r1) + xx_
        h1 = tanh(preactx1)

        # h1 is update1 * h_ + (1 - update1) * h1
        # Leaky integration
        h1 = u1 * h_ + (1. - u1) * h1
        # Multiply h1 by the mask
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        ###########
        # Attention
        ###########
        # h1 X W_comb_att
        pstate_ = tensor.dot(h1, W_comb_att)
        # Accumulate in pctx__
        pctx__ = pctx_ + pstate_[None, :, :]

        # Apply tanh over pctx__
        pctx__ = tanh(pctx__)

        # Affine transformation for alpha = (pctx__ X U_att) + c_att
        alpha = tensor.dot(pctx__, U_att) + c_att

        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])

        # Exponentiate alpha
        alpha = tensor.exp(alpha)

        # If there is a context mask, multiply with it
        # to cancel unnecessary steps
        if context_mask:
            alpha = alpha * context_mask

        # Normalize so that the sum makes 1
        alpha = alpha / (alpha.sum(0, keepdims=True) + 1e-6)

        # Compute the current context ctx_
        # which is cc_ weighted by alpha
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        # Affine transformation: h1 X U_nl + b_nl
        preact2 = tensor.dot(h1, U_nl) + b_nl
        # Add (context X Wc) over it
        preact2 += tensor.dot(ctx_, Wc)
        # Apply sigmoid nonlinearity
        preact2 = tensor.nnet.sigmoid(preact2)

        # Slice activations
        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)
        h2 = tanh(preactx2)

        # Same leaky integration as above now for h2
        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

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
        # NOTE: According to the SAT paper, initial ctx is learned with an FF as well
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] + shared_vars))
    else:
        outputs_info=[init_state,
                      tensor.alloc(0., n_samples, context.shape[2]), # hidden dim
                      tensor.alloc(0., n_samples, context.shape[0])] # n_timesteps

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
