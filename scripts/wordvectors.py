#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np

from sklearn.decomposition import PCA

from collections import OrderedDict

# Word vector manifold à la
# https://github.com/kyunghyuncho/WordVectorManifold/blob/master/LocalChart.ipynb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='wordvectors')
    parser.add_argument('-m', '--model'         , help="Model .npz file.")
    parser.add_argument('-w', '--words'         , help="List of words to analyze.", type=str, nargs='+', required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--target'        , action='store_true',
                                                  help="Work on target level embeddings")
    group.add_argument('-s', '--source'        , action='store_true',
                                                  help="Work on source level embeddings")

    args = parser.parse_args()

    npy = np.load(args.model)
    params = npy['tparams'].tolist()
    opts = dict(npy['opts'])

    if args.target:
        embs = params['Wemb_dec']
        worddict = opts['trg_dict']
        if opts.get('n_words_trg', 0) > 0:
            # Limit dictionary
            worddict = OrderedDict([(k,v) for k,v in worddict.iteritems() if v < opts['n_words_trg']])
    elif args.source:
        embs = params['Wemb_enc']
        worddict = opts['src_dict']
        if opts.get('n_words_src', 0) > 0:
            # Limit dictionary
            worddict = OrderedDict([(k,v) for k,v in worddict.iteritems() if v < opts['n_words_src']])

    print "Number of words in the vocabulary: ", len(worddict)
    print "Embedding dim: %d" % embs.shape[1]

    k = 100
    pca = PCA(n_components=2)
    words = np.array(worddict.keys())

    # get embedding vector
    for w in args.words:
        print '  --> %s' % w
        v = embs[worddict[w]]

        # Find k-NN
        idx = np.argsort(((embs - v[None, :])**2).sum(1))[:k]
        wl = words[idx]
        vl = embs[idx, :]

        # Apply PCA to reduce to 2 dim
        vl_pca = pca.fit_transform(vl)

        # For each axis
        M = 10
        for dix in xrange(vl_pca.shape[1]):
            loc = vl_pca[:, dix] - vl_pca[0, dix]
            dis = loc ** 2
            iid = np.argsort(dis)[:M]
            print ('[dim %d] ' % dix), wl[iid][np.argsort(loc[iid])]
