#!/bin/bash

# Make sure that the following scripts are in your $PATH
#   lowercase.perl
#   tokenizer.perl
#   normalize-punctuation.perl
#   clean-corpus-n-frac.perl

# Create a symlink in your HOME to the data folders
# so that the script works on every machine
ROOT=~/wmt16/data/raw-text/task1

SL="en"
TL="de"
MAX="50"

# For train/val, src/trg languages
for TYPE in train val; do
  OUT=${TYPE}.norm
  for l in $SL $TL; do
    PREFIX="${ROOT}/$TYPE"
    FNAME="${PREFIX}.$l"
    echo $FNAME
    cat $FNAME | normalize-punctuation.perl -l $l > $OUT.$l
    cat $OUT.$l | tokenizer.perl -l $l -threads 8 > tmp_${TYPE}.$l
  done

  # Filter corpus
  clean-corpus-n-frac.perl -ratio 3 tmp_${TYPE} $SL $TL ${OUT}.tok 1 $MAX ${TYPE}.lines
  cat ${OUT}.tok.$SL | lowercase.perl > ${OUT}.lc.tok.$SL
  cat ${OUT}.tok.$TL | lowercase.perl > ${OUT}.lc.tok.$TL
done

rm tmp_*
nmt-build-dict *train*lc.tok.$TL *train*lc.tok.$SL
