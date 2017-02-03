#!/bin/bash
OUT=$1

if [ -z $OUT ]; then
  echo "Usage: $0 <output-dir>"
  exit 1
fi

mkdir $OUT &> /dev/null

for lang in en de; do
  for f in $(ls --color=none data/*$lang); do
    fname=`basename $f`
    fname=${fname/\.$lang/}
    echo "Normalizing punctuation and tokenizing $f"
    cat $f | normalize-punctuation.perl -l $lang | tokenizer.perl -threads 8 -l $lang > $OUT/"$fname.norm.tok.$lang"
  done
done
