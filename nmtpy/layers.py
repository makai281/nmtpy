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
                #'lstm'      : ('param_init_lstm', 'lstm_layer'),
                #'lstm_cond' : ('param_init_lstm_cond', 'lstm_cond_layer'),
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

    # attention:
    params[_p(prefix, 'U_att')]         = norm_weight(dimctx, 1)
    params[_p(prefix, 'c_att')]          = np.zeros((1,)).astype('float32')

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
        n_samples = state_below.shape[1]
    else:
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
    def _step(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
              # The followings are the shared variables
              U, Wc, W_comb_att, U_att, c_att, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]

        pctx__ = tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att) + c_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.nnet.softmax(alpha)

        # Apply context mask over the alphas
        if context_mask:
            alpha = alpha * context_mask

        # Compute the current context
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        preact2 = tensor.dot(h1, U_nl)+b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tanh(preactx2)

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
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]), # hidden dim
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])], # n_timesteps
                                    non_sequences=[pctx_, context] + shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    mode=mode,
                                    strict=True)
    return rval


#### LSTM
#### From: github.com/kelvinxu/arctic-captions

#def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None):
    #"""
     #Stack the weight matrices for all the gates
     #for much cleaner code and slightly faster dot-prods
    #"""
    ## input weights
    #W = np.concatenate([norm_weight(nin,dim),
                           #norm_weight(nin,dim),
                           #norm_weight(nin,dim),
                           #norm_weight(nin,dim)], axis=1)
    #params[_p(prefix,'W')] = W

    ## for the previous hidden activation
    #U = np.concatenate([ortho_weight(dim),
                           #ortho_weight(dim),
                           #ortho_weight(dim),
                           #ortho_weight(dim)], axis=1)
    #params[_p(prefix,'U')] = U

    ## FIXME: Set LSTM bias to 1
    #params[_p(prefix,'b')] = np.zeros((4 * dim,)).astype('float32')

    #return params

## This function implements the lstm fprop
#def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, **kwargs):
    #nsteps = state_below.shape[0]
    #dim = tparams[_p(prefix,'U')].shape[0]

    ## if we are dealing with a mini-batch
    #if state_below.ndim == 3:
        #n_samples = state_below.shape[1]
        #init_state = tensor.alloc(0., n_samples, dim)
        #init_memory = tensor.alloc(0., n_samples, dim)
    ## during sampling
    #else:
        #n_samples = 1
        #init_state = tensor.alloc(0., dim)
        #init_memory = tensor.alloc(0., dim)

    ## if we have no mask, we assume all the inputs are valid
    #if mask == None:
        #mask = tensor.alloc(1., state_below.shape[0], 1)

    ## use the slice to calculate all the different gates
    #def _slice(_x, n, dim):
        #if _x.ndim == 3:
            #return _x[:, :, n*dim:(n+1)*dim]
        #elif _x.ndim == 2:
            #return _x[:, n*dim:(n+1)*dim]
        #return _x[n*dim:(n+1)*dim]

    ## one time step of the lstm
    #def _step(m_, x_, h_, c_):
        #preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        #preact += x_

        #i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        #f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        #o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        #c = tanh(_slice(preact, 3, dim))

        #c = f * c_ + i * c
        #h = o * tanh(c)

        #return h, c, i, f, o, preact

    #state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    #rval, updates = theano.scan(_step,
                                #sequences=[mask, state_below],
                                #outputs_info=[init_state, init_memory, None, None, None, None],
                                #name=_p(prefix, '_layers'),
                                #n_steps=nsteps, profile=options['profile'])
    #return rval

## Conditional LSTM layer with Attention
#def param_init_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None):
    #if nin is None:
        #nin = options['rnn_dim']
    #if dim is None:
        #dim = options['rnn_dim']
    #if dimctx is None:
        #dimctx = options['rnn_dim']

    ## input to LSTM, similar to the above, we stack the matrices for compactness, do one
    ## dot product, and use the slice function below to get the activations for each "gate"
    #W = np.concatenate([norm_weight(nin,dim),
                        #norm_weight(nin,dim),
                        #norm_weight(nin,dim),
                        #norm_weight(nin,dim)], axis=1)
    #params[_p(prefix,'W')] = W

    ## LSTM to LSTM
    #U = np.concatenate([ortho_weight(dim),
                        #ortho_weight(dim),
                        #ortho_weight(dim),
                        #ortho_weight(dim)], axis=1)
    #params[_p(prefix,'U')] = U

    ## bias to LSTM
    ## FIXME: Set LSTM bias to 1
    #params[_p(prefix,'b')] = np.zeros((4 * dim,)).astype('float32')

    ## context to LSTM
    ## NOTE: Why *4?
    #Wc = norm_weight(dimctx,dim*4)
    #params[_p(prefix,'Wc')] = Wc

    ## attention: context -> hidden
    #Wc_att = norm_weight(dimctx, ortho=False)
    #params[_p(prefix,'Wc_att')] = Wc_att

    ## attention: LSTM -> hidden
    #Wd_att = norm_weight(dim,dimctx)
    #params[_p(prefix,'Wd_att')] = Wd_att

    ## attention: hidden bias
    #b_att = np.zeros((dimctx,)).astype('float32')
    #params[_p(prefix,'b_att')] = b_att

    ## optional "deep" attention
    ## Multiple layers
    #if options['n_layers_att'] > 1:
        #for lidx in xrange(1, options['n_layers_att']):
            #params[_p(prefix,'W_att_%d' % lidx)] = ortho_weight(dimctx)
            #params[_p(prefix,'b_att_%d' % lidx)] = np.zeros((dimctx,)).astype('float32')

    ## attention:
    #U_att = norm_weight(dimctx, 1)
    #params[_p(prefix,'U_att')] = U_att

    #c_att = np.zeros((1,)).astype('float32')
    #params[_p(prefix, 'c_att')] = c_att

    #if options['selector']:
        ## attention: selector
        #W_sel = norm_weight(dim, 1)
        #params[_p(prefix, 'W_sel')] = W_sel
        #b_sel = np.float32(0.)
        #params[_p(prefix, 'b_sel')] = b_sel

    #return params

#def lstm_cond_layer(tparams, state_below, options, prefix='lstm',
                    #mask=None, context=None, one_step=False,
                    #init_memory=None, init_state=None,
                    #trng=None, use_dropout=None, sampling=True,
                    #argmax=False, **kwargs):

    #assert context, 'Context must be provided'

    #if one_step:
        #assert init_memory, 'previous memory must be provided'
        #assert init_state, 'previous state must be provided'

    #nsteps = state_below.shape[0]
    #if state_below.ndim == 3:
        #n_samples = state_below.shape[1]
    #else:
        #n_samples = 1

    ## mask
    #if mask is None:
        #mask = tensor.alloc(1., state_below.shape[0], 1)

    ## infer lstm dimension
    #dim = tparams[_p(prefix, 'U')].shape[0]

    ## initial/previous state
    #if init_state is None:
        #init_state = tensor.alloc(0., n_samples, dim)
    ## initial/previous memory
    #if init_memory is None:
        #init_memory = tensor.alloc(0., n_samples, dim)

    ## projected context
    #pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix, 'b_att')]
    #if options['n_layers_att'] > 1:
        #for lidx in xrange(1, options['n_layers_att']):
            #pctx_ = tensor.dot(pctx_, tparams[_p(prefix,'W_att_%d'%lidx)])+tparams[_p(prefix, 'b_att_%d'%lidx)]
            ## note to self: this used to be options['n_layers_att'] - 1, so no extra non-linearity if n_layers_att < 3
            #if lidx < options['n_layers_att']:
                #pctx_ = tanh(pctx_)

    ## projected x
    ## state_below is timesteps*num samples by d in training (TODO change to notation of paper)
    ## this is n * d during sampling
    #state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    ## additional parameters for stochastic hard attention
    #if options['attn_type'] == 'stochastic':
        ## temperature for softmax
        #temperature = options.get("temperature", 1)
        ## [see (Section 4.1): Stochastic "Hard" Attention]
        #semi_sampling_p = options.get("semi_sampling_p", 0.5)
        #temperature_c = theano.shared(np.float32(temperature), name='temperature_c')
        #h_sampling_mask = trng.binomial((1,), p=semi_sampling_p, n=1, dtype=theano.config.floatX).sum()

    #def _slice(_x, n, dim):
        #if _x.ndim == 3:
            #return _x[:, :, n*dim:(n+1)*dim]
        #return _x[:, n*dim:(n+1)*dim]

    #def _step(m_, x_, h_, c_, a_, as_, ct_, pctx_, dp_=None, dp_att_=None):
        #""" Each variable is one time slice of the LSTM
        #m_ - (mask), x_- (previous word), h_- (hidden state), c_- (lstm memory),
        #a_ - (alpha distribution [eq (5)]), as_- (sample from alpha dist), ct_- (context), 
        #pctx_ (projected context), dp_/dp_att_ (dropout masks)
        #"""
        ## attention computation
        ## [described in  equations (4), (5), (6) in
        ## section "3.1.2 Decoder: Long Short Term Memory Network]
        #pstate_ = tensor.dot(h_, tparams[_p(prefix,'Wd_att')])
        #pctx_ = pctx_ + pstate_[:,None,:]
        #pctx_list = []
        #pctx_list.append(pctx_)
        #pctx_ = tanh(pctx_)
        #alpha = tensor.dot(pctx_, tparams[_p(prefix,'U_att')])+tparams[_p(prefix, 'c_att')]
        #alpha_pre = alpha
        #alpha_shp = alpha.shape

        #if options['attn_type'] == 'deterministic':
            #alpha = tensor.nnet.softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
            #ctx_ = (context * alpha[:,:,None]).sum(1) # current context
            #alpha_sample = alpha # you can return something else reasonable here to debug
        #else:
            #alpha = tensor.nnet.softmax(temperature_c*alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
            ## TODO return alpha_sample
            #if sampling:
                #alpha_sample = h_sampling_mask * trng.multinomial(pvals=alpha,dtype=theano.config.floatX)\
                               #+ (1.-h_sampling_mask) * alpha
            #else:
                #if argmax:
                    #alpha_sample = tensor.cast(tensor.eq(tensor.arange(alpha_shp[1])[None,:],
                                               #tensor.argmax(alpha,axis=1,keepdims=True)), theano.config.floatX)
                #else:
                    #alpha_sample = alpha
            #ctx_ = (context * alpha_sample[:,:,None]).sum(1) # current context

        #if options['selector']:
            #sel_ = tensor.nnet.sigmoid(tensor.dot(h_, tparams[_p(prefix, 'W_sel')])+tparams[_p(prefix,'b_sel')])
            #sel_ = sel_.reshape([sel_.shape[0]])
            #ctx_ = sel_[:,None] * ctx_

        #preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        #preact += x_
        #preact += tensor.dot(ctx_, tparams[_p(prefix, 'Wc')])

        ## Recover the activations to the lstm gates
        ## [equation (1)]
        #i = _slice(preact, 0, dim)
        #f = _slice(preact, 1, dim)
        #o = _slice(preact, 2, dim)
        #if options['use_dropout_lstm']:
            #i = i * _slice(dp_, 0, dim)
            #f = f * _slice(dp_, 1, dim)
            #o = o * _slice(dp_, 2, dim)
        #i = tensor.nnet.sigmoid(i)
        #f = tensor.nnet.sigmoid(f)
        #o = tensor.nnet.sigmoid(o)
        #c = tanh(_slice(preact, 3, dim))

        ## compute the new memory/hidden state
        ## if the mask is 0, just copy the previous state
        #c = f * c_ + i * c
        #c = m_[:,None] * c + (1. - m_)[:,None] * c_ 

        #h = o * tanh(c)
        #h = m_[:,None] * h + (1. - m_)[:,None] * h_

        #rval = [h, c, alpha, alpha_sample, ctx_]
        #if options['selector']:
            #rval += [sel_]
        #rval += [pstate_, pctx_, i, f, o, preact, alpha_pre]+pctx_list
        #return rval

    #if options['use_dropout_lstm']:
        #if options['selector']:
            #_step0 = lambda m_, x_, dp_, h_, c_, a_, as_, ct_, sel_, pctx_: \
                            #_step(m_, x_, h_, c_, a_, as_, ct_, pctx_, dp_)
        #else:
            #_step0 = lambda m_, x_, dp_, h_, c_, a_, as_, ct_, pctx_: \
                            #_step(m_, x_, h_, c_, a_, as_, ct_, pctx_, dp_)
        #dp_shape = state_below.shape
        #if one_step:
            #dp_mask = tensor.switch(use_dropout,
                                    #trng.binomial((dp_shape[0], 3*dim),
                                                  #p=0.5, n=1, dtype=state_below.dtype),
                                    #tensor.alloc(0.5, dp_shape[0], 3 * dim))
        #else:
            #dp_mask = tensor.switch(use_dropout,
                                    #trng.binomial((dp_shape[0], dp_shape[1], 3*dim),
                                                  #p=0.5, n=1, dtype=state_below.dtype),
                                    #tensor.alloc(0.5, dp_shape[0], dp_shape[1], 3*dim))
    #else:
        #if options['selector']:
            #_step0 = lambda m_, x_, h_, c_, a_, as_, ct_, sel_, pctx_: _step(m_, x_, h_, c_, a_, as_, ct_, pctx_)
        #else:
            #_step0 = lambda m_, x_, h_, c_, a_, as_, ct_, pctx_: _step(m_, x_, h_, c_, a_, as_, ct_, pctx_)

    #if one_step:
        #if options['use_dropout_lstm']:
            #if options['selector']:
                #rval = _step0(mask, state_below, dp_mask, init_state, init_memory, None, None, None, None, pctx_)
            #else:
                #rval = _step0(mask, state_below, dp_mask, init_state, init_memory, None, None, None, pctx_)
        #else:
            #if options['selector']:
                #rval = _step0(mask, state_below, init_state, init_memory, None, None, None, None, pctx_)
            #else:
                #rval = _step0(mask, state_below, init_state, init_memory, None, None, None, pctx_)
        #return rval
    #else:
        #seqs = [mask, state_below]
        #if options['use_dropout_lstm']:
            #seqs += [dp_mask]
        #outputs_info = [init_state,
                        #init_memory,
                        #tensor.alloc(0., n_samples, pctx_.shape[1]),
                        #tensor.alloc(0., n_samples, pctx_.shape[1]),
                        #tensor.alloc(0., n_samples, context.shape[2])]
        #if options['selector']:
            #outputs_info += [tensor.alloc(0., n_samples)]
        #outputs_info += [None,
                         #None,
                         #None,
                         #None,
                         #None,
                         #None,
                         #None] + [None] # *options['n_layers_att']
        #rval, updates = theano.scan(_step0,
                                    #sequences=seqs,
                                    #outputs_info=outputs_info,
                                    #non_sequences=[pctx_],
                                    #name=_p(prefix, '_layers'),
                                    #n_steps=nsteps, profile=options['profile'])
        #return rval, updates



