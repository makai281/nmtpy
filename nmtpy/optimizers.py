'''
SGD and variants.
'''
import numpy as np

import theano
import theano.tensor as tensor

from itertools import izip as zip

from .nmtutils import itemlist

def get_shared_grads(tparams):
    """Returns shared theano variables of 0 for each param to keep grads."""
    return [theano.shared(p.get_value() * np.float32(0.), name='%s_grad' % k) \
                for k, p in tparams.iteritems()]

def sgd(lr, tparams, grads, inp, cost, profile=False, mode=None):
    """Stochastic Gradient Descent optimizer."""
    gshared = get_shared_grads(tparams)
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # compile theano function to compute cost and copy gradients
    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile, mode=mode)

    # define the update step rule
    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]

    # Compile update rule
    f_update = theano.function([lr], [], updates=pup, profile=profile, mode=mode)
    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost, decay=0.95, profile=False, mode=None):
    """RMSProp optimizer."""

    # Theano tuples for statistics
    gshared     = get_shared_grads(tparams)
    gsup        = [(zg, g) for zg, g in zip(gshared, grads)]
    # Running sum of gradients
    rgrads      = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad' % k)
                    for k, p in tparams.iteritems()]
    # Running sum of squared gradients
    rgrads2     = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad2' % k)
                    for k, p in tparams.iteritems()]
    updir       = [theano.shared(p.get_value() * np.float32(0.), name='%s_updir' % k)
                    for k, p in tparams.iteritems()]

    # don't compute this over and over
    decay_m1    = 1 - decay

    rgup        = [(rg,  decay * rg  + decay_m1 * g)         for rg,  g in zip(rgrads, grads)]
    rg2up       = [(rg2, decay * rg2 + decay_m1 * (g ** 2))  for rg2, g in zip(rgrads2, grads)]

    # compile theano function to compute cost and copy gradients
    f_grad_shared = theano.function(inp, cost, updates=gsup+rgup+rg2up, profile=profile, mode=mode)

    # FIXME: This seems really wrong..
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                    for ud, zg, rg, rg2 in zip(updir, gshared, rgrads, rgrads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]

    # Compile update rule
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=profile, mode=mode)
    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, inp, cost, rho=0.95, eps=1e-6, profile=False, mode=None):
    """Adadelta optimizer."""
    gshared = get_shared_grads(tparams)
    gsup = [(zg, g) for zg, g in zip(gshared, grads)]

    running_up2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rup2' % k)
                    for k, p in tparams.iteritems()]
    # Running sum of squared gradients
    rgrads2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad2' % k)
                for k, p in tparams.iteritems()]

    rho_m1  = 1 - rho
    rg2up   = [(rg2, rho * rg2 + rho_m1 * (g ** 2)) for rg2, g in zip(rgrads2, grads)]

    # compile theano function to compute cost and copy gradients
    f_grad_shared = theano.function(inp, cost, updates=gsup+rg2up, profile=profile, mode=mode)

    updir = [-tensor.sqrt(ru2 + eps) / tensor.sqrt(rg2 + eps) * zg
             for zg, ru2, rg2 in zip(gshared,
                                     running_up2,
                                     rgrads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    # Compile update rule
    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=profile, mode=mode)
    return f_grad_shared, f_update

def adam(lr, tparams, grads, inp, cost, profile=False, mode=None):
    """ADAM optimizer."""
    lr0 = 0.0002
    b1  = 0.9
    b2  = 0.999
    eps = 1e-8

    i = theano.shared(np.float32(0.))
    i_t = i + 1.

    bias_cor_1 = 1. - b1**(i_t)
    bias_cor_2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(bias_cor_2) / bias_cor_1)

    updates = []

    gshared = get_shared_grads(tparams)
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # compile theano function to compute cost and copy gradients
    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile, mode=mode)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)

        m_t = (b1 * m) + ((1. - b1) * g)
        updates.append((m, m_t))

        v_t = (b2 * v) + ((1. - b2) * tensor.sqr(g))
        updates.append((v, v_t))

        p_t = p - (lr_t * (m_t / (tensor.sqrt(v_t) + eps)))
        updates.append((p, p_t))

    updates.append((i, i_t))

    # Compile update rule
    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore', profile=profile, mode=mode)
    return f_grad_shared, f_update
