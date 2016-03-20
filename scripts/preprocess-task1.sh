#!/bin/bash

# Make sure that the following scripts are in your $PATH
#   lowercase.perl
#   tokenizer.perl
#   normalize-punctuation.perl
#   clean-corpus-n-frac.perl

ROOT=/lium/trad4a/wmt/2016/data
BDIC=/lium/buster1/caglayan/git/nmtpy/bin/nmt-build-dict
SELECT_LINES=/lium/buster1/caglayan/bin/select-lines
SL="en"
TL="de"
MAX="50"

# For train/val, src/trg languages
# lowercase->normalize->strip punctuation->tokenize
for TYPE in train valid; do
  OUT=${TYPE}.norm
  for l in $SL $TL; do
    PREFIX="${ROOT}/text-$TYPE/$TYPE"
    FNAME="${PREFIX}.$l"
    echo $FNAME
    cat $FNAME | normalize-punctuation.perl -l $l > $OUT.$l
    cat $OUT.$l | tokenizer.perl -l $l -threads 8 > tmp_${TYPE}.$l
  done

  # Filter corpus
  clean-corpus-n-frac.perl tmp_${TYPE} $SL $TL ${OUT}.tok 1 $MAX 3 tmp_${TYPE}.lines.retained
  cat ${OUT}.tok.$SL | lowercase.perl > ${OUT}.lc.tok.$SL
  cat ${OUT}.tok.$TL | lowercase.perl > ${OUT}.lc.tok.$TL
done

## Lium's split
for l in $SL $TL; do
  $SELECT_LINES tmp_train.$l lium_train.id 1 > lium_train.norm.tok.$l
  cat lium_train.norm.tok.$l | lowercase.perl > lium_train.norm.lc.tok.$l
done

rm tmp_*
$BDIC *train*lc.tok.$TL *train*tok.$SL
