#!/usr/bin/env python

import sys
import numpy as np

import seaborn as sns
import pandas as pd

from collections import OrderedDict

def parse_log(fname):
    """Parses a log file and returns an info dict if the training is completed."""

    lines = []
    epoch_losses = []
    valid_losses = []
    valid_bleus  = []
    d = OrderedDict()
    with open(fname) as f:
        for line in f:
            lines.append(line.strip().split(' ', 2)[-1])

    for line in lines:
        if " -> " in line:
            parts = line.split()
            d[parts[0]] = parts[-1]
        elif "finished with mean batch loss" in line:
            epoch_losses.append(float(line.split(":")[-1].strip()))
        elif "] BLEU" in line:
            bleu = line.split("BLEU = ")[-1]
            valid_bleus.append(float(bleu.split(",")[0]))
        elif "] LOSS" in line:
            sp = -1
            if "PX" in line:
                sp = -3
            valid_losses.append(float(line.split()[sp]))

    for line in lines[-100:]:
        if ", update: " in line:
            d['last_update'] = int(line.split(",")[-3].split()[-1])
        elif "Early Stopped" in line:
            d['finished'] = True
            break

    if "finished" not in d:
        d['finished'] = False

    if d['valid_freq'] == "0":
        d['valid_freq'] = "epochs"

    d['decay_c'] = float(d['decay_c'])
    d['alpha_c'] = float(d['alpha_c'])

    d['train_loss'] = epoch_losses
    d['valid_loss'] = valid_losses
    d['valid_bleu'] = valid_bleus
    d['Epochs'] =  len(epoch_losses)

    best_bleu_idx = np.array(d['valid_bleu']).argmax()
    d['best_metric_validx'] = best_bleu_idx

    best_vloss_idx = np.array(d['valid_loss']).argmin()
    d['best_vloss'] = (best_vloss_idx, d['valid_loss'][best_vloss_idx])
    d['Valid PPL'] = np.exp(d['best_vloss'][1])

    return d

def plot(result, title):
    trainloss, validloss, validbleu = result
    miniter = min(len(trainloss), len(validloss))
    trainloss = np.array(trainloss[:miniter])
    validloss = np.array(validloss[:miniter])
    validbleu = np.array(validbleu)
    max_bleu = valid_bleu.max()

    f, (ax1, ax2) = sns.plt.subplots(2, 1, sharex=True)
    ax1.set_title(title + ' (BLEU: %.2f)' % max_bleu)
    ax1.plot(trainloss, label='TRAIN LOSS')
    ax1.plot(validloss, label='VALID LOSS')
    ax2.plot(validbleu, label='VALID BLEU', color='red')
    ax1.legend()
    ax1.legend(loc='lower right')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print 'Usage: %s <model log files>' % sys.argv[0]
        sys.exit(1)

    models = sys.argv[1:]
    results = OrderedDict()
    # Parse log files
    for model in models:
        log = parse_log(model)
        results[model] = (log['train_loss'], log['valid_loss'], log['valid_bleu'])
