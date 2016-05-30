#!/usr/bin/env python

import os
import sys
import cPickle
import subprocess
from datetime import datetime

# Evaluate on both valid and test split using nmt-coco-metrics
def evaluate_metrics(hyps_file, ref_list, language):
    res = subprocess.check_output(['nmt-coco-metrics', '-l', language, hyps_file] + ref_list)
    return eval(res)

def gen_report(log_file, outs_dir):
    attributes = ['alpha_c', 'batch_size', 'clip_c', 'decay_c',
                  'dropout', 'lrate', 'optimizer', 'model_type',
                  'patience','rnn_dim', 'trg_emb_dim', 'valid_freq',
                  'valid_metric', 'weight_init']

    time_format = '%Y-%m-%d %H:%M:%S'

    report = {}
    vals = []
    last_epoch = 0
    last_update = 0

    with open(log_file) as f:
        # Get start time
        start_time = datetime.strptime(f.readline().strip().split(",")[0], time_format)

        # Read other lines
        for line in f:
            line = line.strip()
            # Configuration parameter
            if " -> " in line:
                fields = line.split(" ")
                key, value = fields[-3], fields[-1]
                if key in attributes:
                    report[key] = value
            elif "[Validation " in line:
                vals.append(line)
            elif "Epoch:" in line:
                fields = line.split(",")
                last_epoch = fields[1].split(" ")[-1]
                last_update = fields[2].split(" ")[-1]

        stop_time = datetime.strptime(line.split(",", 1)[0], time_format)

    # Number of updates and epoch during training
    report['nb_updates'] = last_update
    report['nb_epochs'] = last_epoch

    report['best_val_update'] = vals[-1].split("]")[0].split(" ")[-1]
    report['best_val_metric'] = vals[-1].split("Best ")[-1]

    for line in vals:
        if "%s] LOSS" % report['best_val_update'] in line:
            report['best_val_loss'] = line.split(" ")[-1]

    # Training duration
    report['duration'] = stop_time - start_time

    if "weight_init" not in report:
        # This was default when this parameter wasn't around
        report['weight_init'] = '0.01'

    # lrate not effective when optimizer is not sgd for now
    if report['optimizer'] != 'sgd':
        del report['lrate']

    return report

def translate(npz_file, outs_dir, beamsize=12, src_file=None):
    split = "dev" if src_file is None else "test"
    model = os.path.basename(npz_file)
    dirname = os.path.dirname(npz_file)
    hyp_file = os.path.join(outs_dir, "%s.beam%d.%s.1best" % (model, beamsize, split))

    if not os.path.exists(hyp_file):
        cmd = ['nmt-translate', '-m', npz_file, '-b', str(beamsize), '-o', hyp_file]
        if src_file:
            cmd.extend(['-S', src_file])

        print "Translating '%s' with beamsize: %d" % (split, beamsize)
        result = subprocess.check_output(cmd)
        print "Done."
    else:
        print "%s already exists, skipping."
    return hyp_file

def process_model(log_file):
    dirname = os.path.dirname(log_file)
    outs_dir = os.path.join(dirname, 'outs')

    if not os.path.exists(outs_dir):
        os.mkdir(outs_dir)

    base_file = os.path.basename(log_file)
    model = os.path.splitext(base_file)[0]
    npz = os.path.join(dirname, model + ".npz")
    pkl = os.path.join(dirname, npz + ".pkl")

    assert os.path.exists(npz)
    assert os.path.exists(pkl)

    # Collect information about the model
    report = gen_report(log_file, outs_dir)



if __name__ == '__main__':
    log_file = sys.argv[1]
