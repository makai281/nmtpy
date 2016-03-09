#!/bin/bash

if [ -z $1 ]; then
  echo "Usage: $0 <output_directory>"
  exit 1
fi

ODIR=$1
mkdir $ODIR

# Languages
SRCL=en
TRGL=de

# Scripts from subword github
APPLY=/lium/buster1/caglayan/bin/apply_bpe.py
LEARN=/lium/buster1/caglayan/bin/learn_bpe.py
BDICT=/lium/buster1/caglayan/git/theano-attention/scripts/build_dictionary.py

# Moses tokenized corpora files
TOK_PATH=/lium/trad4a/wmt/2016/caglayan/theano-attention/data/moses.tok/tok

# Prefix
TRAIN=train.moses.tok
VALID=val.moses.tok

# Concatenate everything
cat $TOK_PATH/{$TRAIN.$SRCL,$TRAIN.$TRGL,$VALID.$SRCL,$VALID.$TRGL} > $ODIR/all.joint.tok

set -x
# Prepare BPE files for 6 different values and apply them to the corpora
for s in 1000 5000 10000 20000 30000 40000; do
  if [ ! -f $ODIR/all.joint.tok.s$s.bpe ]; then
    $LEARN -s $s -i $ODIR/all.joint.tok -o $ODIR/all.joint.tok.s$s.bpe
    $APPLY -c $ODIR/all.joint.tok.s$s.bpe -i $TOK_PATH/$TRAIN.$SRCL -o $ODIR/$TRAIN.subword$s.$SRCL
    $APPLY -c $ODIR/all.joint.tok.s$s.bpe -i $TOK_PATH/$TRAIN.$TRGL -o $ODIR/$TRAIN.subword$s.$TRGL

    $APPLY -c $ODIR/all.joint.tok.s$s.bpe -i $TOK_PATH/$VALID.$SRCL -o $ODIR/$VALID.subword$s.$SRCL
    $APPLY -c $ODIR/all.joint.tok.s$s.bpe -i $TOK_PATH/$VALID.$TRGL -o $ODIR/$VALID.subword$s.$TRGL
    $BDICT -o $ODIR $ODIR/$TRAIN.subword$s.{$SRCL,$TRGL}
  fi
done
set +x
