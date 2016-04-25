#!/usr/bin/env python
import os
import sys
import json
from itertools import izip

if __name__ == '__main__':
    try:
        imglist = sys.argv[1]
        hyps = sys.argv[2]
    except Exception as e:
        print "Usage: %s <image list> <hyps1> <hyps2> .." % sys.argv[0]
        sys.exit(1)

    with open(imglist) as f:
        imglist = f.read().strip().split("\n")

    systems = []
    hyps = []
    for fname in sys.argv[2:]:
        print "Reading %s" % fname
        systems.append(os.path.basename(fname).split(".")[0])

        with open(fname) as f:
            hyps.append(f.read().strip().split("\n"))

        assert len(hyps[-1]) == len(imglist)

    hyps = zip(*hyps)

    # Find out metrics

    # list of dicts, each element is {'image_id': somenumber, 'caption': list of captions}
    dump = []
    for hyp, img in izip(hyps, imglist):
        dump.append({'image_id' : img, 'sents' : hyp})

    final = {'captions' : dump, 'metadata' : {}}

    with open('vis.json', 'w') as f:
        json.dump(final, f)
