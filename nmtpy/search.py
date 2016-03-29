#!/usr/bin/env python
import copy
import numpy as np

from .typedef import INT

###################################################
def forced_decoding(f_init, f_next, inputs, target):
    # get initial state of decoder rnn and encoder context
    if len(inputs) == 1:
        next_state, ctx0 = f_init(inputs[0])
    else:
        next_state, ctx0 = f_init(*inputs)

    # Beginning-of-sentence indicator
    next_sample = -1 * np.ones((1,)).astype(INT)

    final_sample = []
    final_score = 0
    ctx = np.tile(ctx0, [1, 1])

    for ii in xrange(len(target)):

        # Get next states
        inputs = [next_sample, ctx, next_state]
        next_log_p, next_state = f_next(*inputs)

        sample.append(target[ii])
        final_score += next_log_p[0, target[ii]]

    return final_sample, final_score

def gen_sample(f_init, f_next, inputs, maxlen=50, argmax=False):
    sample = []
    sample_score = 0

    # get initial state of decoder rnn and encoder context
    if len(inputs) == 1:
        next_state, ctx0 = f_init(inputs[0])
    else:
        next_state, ctx0 = f_init(*inputs)

    # Beginning-of-sentence indicator
    next_sample = -1 * np.ones((1,)).astype(INT)

    for ii in xrange(maxlen):
        ctx = np.tile(ctx0, [1, 1])

        # Get next states
        inputs = [next_sample, ctx, next_state]
        next_p, next_sample, next_state = f_next(*inputs)

        if argmax:
            nw = next_p[0].argmax()
        else:
            nw = next_sample[0]

        sample.append(nw)
        sample_score += np.log(next_p[0, nw])

        # EOS
        if nw == 0:
            break

    return sample, sample_score
