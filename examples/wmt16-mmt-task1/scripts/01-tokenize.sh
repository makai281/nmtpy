#!/bin/bash

# Make sure that the following scripts are in your $PATH
#   lowercase.perl
#   tokenizer.perl
#   normalize-punctuation.perl
#   clean-corpus-n-ratio.perl
export LC_ALL=en_US.UTF_8

OUT=$1

if [ -z $OUT ]; then
  echo "Usage: $0 <output-dir>"
  exit 1
fi

mkdir -p $OUT &> /dev/null

# Process train and val
for TYPE in "train" "val" "test"; do
  for l in en de; do
    # Normalize, tokenize and save under $OUT
    cat data/${TYPE}.$l | normalize-punctuation.perl -l $l | tokenizer.perl -l $l -threads 8 > $OUT/${TYPE}.norm.tok.$l
  done

  if [ $TYPE == "train" ]; then
    # Filter corpus to remove a few buggy sentences
    clean-corpus-n-ratio.perl -ratio 3 ${OUT}/${TYPE}.norm.tok en de ${OUT}/${TYPE}.norm.max50.tok 2 50 ${OUT}/${TYPE}.lines
    # Lowercase
    cat ${OUT}/${TYPE}.norm.max50.tok.en | lowercase.perl > ${OUT}/${TYPE}.norm.max50.tok.lc.en
    cat ${OUT}/${TYPE}.norm.max50.tok.de | lowercase.perl > ${OUT}/${TYPE}.norm.max50.tok.lc.de
  else
    # Lowercase and save
    cat ${OUT}/${TYPE}.norm.tok.en | lowercase.perl > ${OUT}/${TYPE}.norm.tok.lc.en
    cat ${OUT}/${TYPE}.norm.tok.de | lowercase.perl > ${OUT}/${TYPE}.norm.tok.lc.de
  fi
done
