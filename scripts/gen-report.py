#!/usr/bin/env python

import sys
import os
import cPickle

import pandas as pd

def plot(model):
    # Plot model training_loss, validation_loss, (BLEU in the future)
    pass

def generate_html_report(reports):
    # Generate a table with parameters and results
    # include also plots?
    pass

if __name__ == '__main__':
    # keys are model identifiers, values are model parameters and results
    # parsed from log file
    reports = {}

    for log_file in sys.argv[1:]:
        print "Analyzing %s" % log_file

        model_id = os.path.splitext(os.path.basename(log_file))[0]
        # It is easier to read params from model params pkl file instead of the log file
        model_params = cPickle.load(open("%s.npz.pkl" % model_id, "rb"))

        report = {}

        report['src_vocab_size'] = len(model_params['src_vocab'])
        report['trg_vocab_size'] = len(model_params['trg_vocab'])
        report['optimizer'] = model_params['optimizer']
        report['lrate'] = model_params['lrate']
        report['batch_size'] = model_params['batch_size']
        report['emb_dim'] = model_params['embedding_dim']
        report['gru_dim'] = model_params['gru_dim']
        report['dropout'] = model_params['dropout'] if model_params['dropout'] > 0. else "no"
        report['vgg'] = model_params.get("vgg-feats-file", "no")
        report['maxlen'] = model_params['maxlen']
        report['alpha_c'] = model_params['alpha_c']
        report['decay_c'] = model_params['decay_c']
        report['clip_c'] = model_params['clip_c'] if model_params['clip_c'] > 0. else "no"

        log_lines = open(log_file, "rb").read().strip().split("\n")

        mb_updates = [l for l in log_lines if "Epoch: " in l]
        report['n_updates'] = len(mb_updates)

        train_costs = []
        mean_batch_train_time = 0.0
        for line in mb_updates:
            fields = line.split(" ")
            train_costs.append(float(fields[-3].strip(",")))
            mean_batch_train_time += float(fields[-1].strip())

        # Compute some statistics
        mean_batch_train_time /= len(mb_updates)

        # Downsample train costs (step: 20)
        report['train_costs'] = train_costs[::20]
        report['mean_cost'] = sum(train_costs) / len(train_costs)
        report['mean_batch_train_time'] = mean_batch_train_time

        # Compute validation loss
        val = [float(l.split()[-4]) for l in log_lines if "Validation loss" in l]
        mean_val_loss = sum(val) / len(val)
        report['n_validation'] = len(val)
        report['mean_val_loss'] = mean_val_loss

        # Add to final list
        reports[model_id] = report
