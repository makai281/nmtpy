#!/usr/bin/env python

import os
import sys
import cPickle

import numpy
numpy.random.seed(1234)

IMG_ROOT="/lium/trad4a/wmt/2016/data/flickr30k-images/"
SPLIT_DICT="/lium/trad4a/wmt/2016/data/flickr30k-image-splits/data-split.pkl"

VAL_PREFIX="/lium/trad4a/wmt/2016/caglayan/theano-attention/data/moses.tok/tok/val.moses.tok."

TABLE="""\
<html>
<head>
    <meta charset="UTF-8" />
    <title>Image Comparison</title>
</head>
<body>
    <table border="0">
        <tr>
            <td><b>Model name</b></td><td><b>BLEU score</b></td>
        </tr>
        %s
    </table>
    <table border="0" cellpadding="10px" cellspacing="10px" style="background-color: #ffffff; border: 1px solid #ccc;">
        %s
    </table>
</body>
"""

BLEU="""\
        <tr>
            <td>%s</td><td>%s</td>
        </tr>
"""

ROW="""\
    <tr>
        <td style="padding-top:10px; padding-bottom:10px; padding-left:10px;">
            <img src="data:image/jpeg;base64,%(img)s" class="caption" style="display: block; max-width:250px;" />
        </td>
        <td valign="middle" stype="color:#505050; font-family:Georgia, serif; font-size:12px; line-height:100%%; padding:20px;">
            <table border="0" cellpadding="5px" cellspacing="5px">
                %(trans)s
            </table>
        </td>
    </tr>

"""
CAPTION="""\
            <tr style="color: %s"><td align="right"><b>%s&nbsp;</b></td><td>%s</td></tr>
"""

def get_row(img_path, captions):
    d = {}
    d['img'] = open(img_path).read().encode('base64')
    d['trans'] = "\n".join([CAPTION % (x[2], x[0], x[1]) for x in captions])
    return ROW % d

if __name__ == '__main__':
    trans   = []
    bleus   = []

    next_argc = 1
    out_file = "out.html"
    if sys.argv[1] == '-o':
        out_file = sys.argv[2]
        next_argc += 2

    # Files ending with 1best
    models  = sys.argv[next_argc:]

    for model in models:
        # Each file is the translations produced by an NMT for the val_src set.
        trans.append(open(model).read().strip().split("\n"))
        bleus.append(open(model + ".bleu").read().strip())

    # Cut down directory part from models
    models = [os.path.splitext(os.path.basename(m))[0] for m in models]

    # Sort BLEU of models by decreasing order
    sorted_idxs = [x for x,y in sorted(enumerate(bleus), key=lambda x: x[1], reverse=True)]

    # Infer language pair from model name
    src_lang, trg_lang = models[0].split("-")[1:3]

    # Load source reference sentences
    splits = cPickle.load(open(SPLIT_DICT))
    val_imgs = splits["valid"]

    # choose 10 random images
    idxs = numpy.random.randint(0, len(val_imgs), 100)

    val_src = open(VAL_PREFIX + src_lang).read().strip().split("\n")
    val_ref = open(VAL_PREFIX + trg_lang).read().strip().split("\n")

    assert len(val_src) == len(val_ref) == len(val_imgs)

    content = ""
    for i in idxs:
        model_captions = [('Source', val_src[i], 'blue'), ('Reference', val_ref[i], 'green')]
        for j in sorted_idxs:
            model_captions.append((models[j], trans[j][i], 'black'))
        content += get_row(os.path.join(IMG_ROOT, val_imgs[i] + ".jpg"), model_captions)

    bleu_list = "\n".join([BLEU % (models[i], bleus[i]) for i in sorted_idxs])
    with open(out_file, "wb") as f:
        f.write(TABLE % (bleu_list, content))
