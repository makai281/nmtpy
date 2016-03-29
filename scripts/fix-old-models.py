#!/usr/bin/env python
import sys
import cPickle

def fix_data_dict(d):
    return dict([[k, v[1]] for k,v in d.items()])

if __name__ == '__main__':

    for pkl in sys.argv[1:]:
        with open(pkl) as f:
            d = cPickle.load(f)

        if "trng" in d:
            print "Removing trng object."
            del d["trng"]

        data = d['data']
        for k,v in data.iteritems():
            if isinstance(v, list) and v[0] in ["img_feats", "text", "bitext"]:
                print "Fixing data dict."
                d['data'] = fix_data_dict(data)

        with open(pkl, 'w') as f:
            cPickle.dump(d, f)
