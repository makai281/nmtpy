#!/usr/bin/env python

from itertools import izip
import sys

# Extracts the 1-best of N different hypotheses produced
# by feeding N different source sentences for a single image
# File format is:
# 0 ||| s1-1 ||| score1-1
# ....
# 1023 ||| s1-1024 ||| score1-1024
### Here we restart with different inputs if validation set has 1024 samples
# 1024 ||| s2-1 ||| score2-1

if __name__ == '__main__':
    try:
        n_samples = int(sys.argv[1])
        hyps_file = sys.argv[2]
    except:
        print "Usage: %s <# samples in valid set> <hyps file>" % sys.argv[0]
        sys.exit(1)

    lines = []
    with open(hyps_file) as f:
        for line in f:
            lines.append(line.strip())

    if len(lines) % n_samples != 0:
        print "Error: Number of lines is not a multiple of n_samples!"
        sys.exit(1)
    n_splits = len(lines) / n_samples

    hyps = [[] for i in range(n_splits)]
    for i in range(len(lines)):
        hyps[i / n_samples].append(lines[i].split(" ||| "))

    for hyp in izip(*hyps):
        # Sort by last column which is score and take the minimum one
        best = sorted(hyp, key=lambda x: float(x[-1]))[0]
        print best[1]
