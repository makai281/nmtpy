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

########################################################
def beam_search(f_init, f_next, inputs, beam_size=12, maxlen=50):
    # Final results and their scores
    final_sample = []
    final_score = []

    live_beam = 1
    dead_beam = 0

    # Initially we have one empty hypothesis
    # with a score of 0
    hyp_states  = []
    hyp_samples = [[]]
    hyp_scores  = np.zeros(1).astype('float32')

    # get initial state of decoder rnn and encoder context
    # The check is for multimodal data
    if len(inputs) == 1:
        next_state, ctx0 = f_init(inputs[0])
    else:
        next_state, ctx0 = f_init(*inputs)

    # Beginning-of-sentence indicator is -1
    next_w = -1 * np.ones((1,)).astype(INT)

    # Iterate until maxlen. In groundhog this
    # is len(seq) * 3
    for ii in xrange(maxlen):
        ctx = np.tile(ctx0, [live_beam, 1])

        # Get next states
        inputs = [next_w, ctx, next_state]
        next_log_p, next_state = f_next(*inputs)

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
        new_hyp_scores  = np.zeros(beam_size-dead_beam).astype('float32')
        new_hyp_samples = []

        # Iterate over the hypotheses
        # and add them to new_* lists
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

        if new_live_beam < 1:
            break
        if dead_beam >= beam_size:
            break

        # Prepare for the next iteration
        next_w = np.array([w[-1] for w in hyp_samples])
        next_state = np.array(hyp_states)

    # dump every remaining hypotheses
    if live_beam > 0:
        for idx in xrange(live_beam):
            final_sample.append(hyp_samples[idx])
            final_score.append(hyp_scores[idx])

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
