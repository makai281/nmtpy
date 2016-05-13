#!/usr/bin/env python
import os
import sys
import json
import argparse
from itertools import izip

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imglist', type=str, help='List of image files.')
    parser.add_argument('-s', '--srcfile', type=str, help='Source sentences file.')
    parser.add_argument('-t', '--hypfiles', nargs='+', type=str, help='System output(s).')
    args = parser.parse_args()

    with open(args.imglist) as f:
        imglist = f.read().strip().split("\n")

    with open(args.srcfile) as f:
        src_sents = f.read().strip().split("\n")

    assert len(src_sents) == len(imglist)

    metadata = []
    hyps = []

    for fname in args.hypfiles:
        print "Reading %s" % fname
        sysname = os.path.basename(fname).rsplit(".", 2)[0]
        d = {'name' : sysname}
        try:
            score = open('%s.score' % fname).read().strip()
            d = eval(score)
            d = dict([(k.upper().replace("_", ""), "%.2f" % (100*float(v))) for k,v \
                        in d.iteritems() if k.startswith(("Bleu", "METEOR"))])
        except IOError as ie:
            pass

        metadata.append(d)

        with open(fname) as f:
            hyps.append(f.read().strip().split("\n"))

        assert len(hyps[-1]) == len(imglist)

    hyps = zip(*hyps)

    # Find out metrics

    # list of dicts, each element is {'image_id': somenumber, 'caption': list of captions}
    dump = []
    for ssent, hyp, img in izip(src_sents, hyps, imglist):
        dump.append({'image_id' : img, 'sents' : hyp, 'src' : ssent})

    final = {'captions' : dump, 'metadata' : metadata}

    with open('vis.json', 'w') as f:
        json.dump(final, f)
