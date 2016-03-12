#!/usr/bin/env python
import copy
import numpy as np

from .typedef import INT

try:
    # Pretty fast logarithm with MKL. Makes beam_search
    # faster. Use this if available.
    import numexpr
    logarithm = lambda x: numexpr.evaluate("log(x)")
except ImportError as ie:
    logarithm = lambda x: np.log(x)

def forced_decoding(f_init, f_next, inputs, target):
    # get initial state of decoder rnn and encoder context
    if len(inputs) == 1:
        next_state, ctx0 = f_init(inputs[0])
    else:
        next_state, ctx0 = f_init(*inputs)

    # Beginning-of-sentence indicator
    next_sample = -1 * np.ones((1,)).astype(INT)

    sample = []
    sample_score = 0

    for ii in xrange(len(target)):
        ctx = np.tile(ctx0, [1, 1])

        # Get next states
        inputs = [next_sample, ctx, next_state]
        next_p, next_sample, next_state = f_next(*inputs)

        sample.append(target[ii])
        sample_score += np.log(next_p[0, target[ii]])

    return sample, sample_score

def beam_search(f_init, f_next, inputs, k=1, maxlen=50):
    # Final results and their scores
    sample = []
    sample_score = []

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    if len(inputs) == 1:
        next_state, ctx0 = f_init(inputs[0])
    else:
        next_state, ctx0 = f_init(*inputs)

    # Beginning-of-sentence indicator
    next_w = -1 * np.ones((1,)).astype(INT)

    for ii in xrange(maxlen):
        ctx = np.tile(ctx0, [live_k, 1])

        # Get next states
        inputs = [next_w, ctx, next_state]
        next_p, next_w, next_state = f_next(*inputs)

        # Beam search
        cand_scores = hyp_scores[:, None] - logarithm(next_p)
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]

        voc_size = next_p.shape[1]
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = np.zeros(k-dead_k).astype('float32')
        new_hyp_states = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []

        for idx in xrange(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                # EOS detected
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])

        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = np.array([w[-1] for w in hyp_samples])
        next_state = np.array(hyp_states)

    # dump every remaining hypotheses
    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    return sample, sample_score


################
# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def beam_search2(f_init, f_next, inputs, beam_size=12, maxlen=50):
    # Final results and their scores
    sample = []
    sample_score = []

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    if len(inputs) == 1:
        next_state, ctx0 = f_init(inputs[0])
    else:
        next_state, ctx0 = f_init(*inputs)

    # Beginning-of-sentence indicator
    next_w = -1 * np.ones((1,)).astype(INT)

    for ii in xrange(maxlen):
        ctx = np.tile(ctx0, [live_k, 1])

        # Get next states
        inputs = [next_w, ctx, next_state]
        next_p, next_w, next_state = f_next(*inputs)

        # Beam search
        cand_scores = hyp_scores[:, None] - logarithm(next_p)
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(beam_size-dead_k)]

        voc_size = next_p.shape[1]
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = np.zeros(beam_size-dead_k).astype('float32')
        new_hyp_states = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []

        for idx in xrange(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                # EOS detected
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])

        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= beam_size:
            break

        next_w = np.array([w[-1] for w in hyp_samples])
        next_state = np.array(hyp_states)

    # dump every remaining hypotheses
    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    # Return normalized 1-best hyp idxs
    lens = np.array([len(s) for s in sample])
    sample_score = np.array(sample_score) / lens
    sid = np.argmin(sample_score)
    return sample[sid]

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
