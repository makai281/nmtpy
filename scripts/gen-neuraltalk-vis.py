#!/usr/bin/env python
import os
import sys
import json
import argparse
from itertools import izip

import numpy as np

def escape(s):
    return s.replace("<", "&lt;").replace(">", "&gt;")

def get_hyps(hypfile, n_samples, nbest):
    lines = []
    with open(hypfile) as f:
        for line in f:
            lines.append(line.strip())

    if nbest:
        # n-best list of multiple source predictions
        n_splits = len(lines) / n_samples
        hyps = [[] for i in range(n_splits)]
        for i in range(len(lines)):
            hyps[i / n_samples].append(lines[i].split(" ||| "))

        # Pick the no <unk> one out of N translations
        # Sort by last column which is score and take the minimum one
        final_hyps = []
        for hyp in izip(*hyps):
            sorted_idxs = np.argsort([float(h[2]) for h in hyp])
            no_unks = ["<unk>" not in hyp[i][1] for i in sorted_idxs]
            try:
                idx = sorted_idxs[no_unks.index(True)]
            except ValueError as ve:
                # Every hyp contains <unk>
                idx = sorted_idxs[0]
            finally:
                final_hyps.append((idx, escape(hyp[idx][1])))

        return final_hyps
    else:
        return lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imglist', type=str, help='List of image files.')
    parser.add_argument('-s', '--srcfiles', nargs='+', type=str, help='Source sentences file(s).')
    parser.add_argument('-n', '--sysnames', nargs='+', type=str, help='System name(s).', default=None)
    parser.add_argument('-t', '--hypfiles', nargs='+', type=str, help='System output(s).')
    args = parser.parse_args()

    with open(args.imglist) as f:
        imglist = f.read().strip().split("\n")

    src_sents = []
    for fname in args.srcfiles:
        print "Reading source file: %s" % fname
        with open(fname) as f:
            sents = f.read().strip().split("\n")
            assert len(sents) == len(imglist)
            src_sents.append(sents)

    n_samples = len(src_sents[0])
    print "Number of samples: %d" % n_samples

    nbest = len(src_sents) > 1
    nsplits = len(src_sents)

    src_sents = zip(*src_sents)
    metadata = []
    hyps = []

    test_scores = {'LIUM_2_TextNMT_C'           : {'BLEU4' : 23.8, 'METEOR' : 35.1},
                   'LIUMCVC_2_MultimodalNMT_C'  : {'BLEU4' : 19.2, 'METEOR' : 32.3}}

    for idx, fname in enumerate(args.hypfiles):
        if args.sysnames:
            sysname = args.sysnames[idx]
        else:
            sysname = os.path.basename(fname).rsplit(".", 2)[0]
        d = {'name' : sysname}
        try:
            score = open('%s.score' % fname).read().strip()
            s = eval(score)
            s = dict([(k.upper().replace("_", ""), "%.2f" % (100*float(v))) for k,v \
                        in d.iteritems() if k.startswith(("Bleu", "METEOR"))])
        except IOError as ie:
            s = test_scores[sysname]
        finally:
            d.update(s)

        metadata.append(d)
        hyps.append(get_hyps(fname, n_samples, nbest))

    hyps = zip(*hyps)

    dump = []

    for hyp, img in izip(hyps, imglist):
        if nbest:
            src_idxs = [h[0] for h in hyp]
            hyp = [h[1] for h in hyp]
        else:
            src_idxs = [0 for h in hyp]
        dump.append({'image_id' : img, 'outs' : hyp, 'sidxs' : src_idxs})

    # srcsents is a list of source sentence(s) for each image.
    final = {'srcsents' : src_sents,
             'nsplits'  : nsplits,
             'captions' : dump,
             'metadata' : metadata}

    with open('vis.json', 'w') as f:
        json.dump(final, f)
