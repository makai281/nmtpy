#!/usr/bin/env python

# Compare the log files of multiple runs of the same model

import sys
import numpy as np
from hashlib import sha1

pxs     = []
sums    = []
bleus   = []
epochs  = []

# Collect stats
for model in sys.argv[1:]:
    log = open(model).read().strip().split("\n")
    bl  = []
    px  = []
    eps = []
    for line in log:
        if "[Validation" in line and "] BLEU" in line:
            # validation bleu scores
            bl.append(line.split("]")[-1].strip())
        if "[Validation" in line and "PX: " in line:
            # validation perplexities
            px.append(line.split(":")[-1].replace(")", ""))
        elif "mean batch loss" in line:
            # epoch losses
            eps.append(line.split(":")[-1])

    if eps:
        epochs.append(eps)
    if px:
        pxs.append(px)
    if bl:
        bleus.append(bl)

len_common_pxs    = min([len(x) for x in pxs])
len_common_bleus  = min([len(x) for x in bleus])
len_common_epochs = min([len(x) for x in epochs])

pxs    = [b[:len_common_pxs] for b in pxs]
bleus  = [b[:len_common_bleus] for b in bleus]
epochs = [b[:len_common_epochs] for b in epochs]

# Compare
result = 0
result += len(set([sha1("\n".join(b)).hexdigest() for b in bleus])) > 1
result += len(set([sha1("\n".join(b)).hexdigest() for b in epochs])) > 1
result += len(set([sha1("\n".join(b)).hexdigest() for b in pxs])) > 1

if result > 0:
    print 'FAIL.'
else:
    print 'OK.'
