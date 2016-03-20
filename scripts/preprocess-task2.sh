#!/bin/bash

# Make sure that the following scripts are in your $PATH
#   lowercase.perl
#   tokenizer.perl
#   normalize-punctuation.perl
#   clean-corpus-n-frac.perl

ROOT=/lium/trad4a/wmt/2016/data/crosslingual-task
BDIC=/lium/buster1/caglayan/git/nmtpy/bin/nmt-build-dict
SELECT_LINES=/lium/buster1/caglayan/bin/select-lines
SL="en"
TL="de"
MAX="50"

# Put everything under this directory
mkdir -p $SL/{train,val} $TL/{train,val} &> /dev/null

# For train/val, src/trg languages
# lowercase->normalize->strip punctuation->tokenize
for TYPE in train val; do
  for l in $SL $TL; do
    for f in `ls $ROOT/$l/$TYPE/*`; do
      FNAME=`basename $f`
      OUT="$l/$TYPE/$FNAME.lc"
      if [ ! -f "$OUT.norm.nopunct.tok.$l" ]; then
        echo $f
        cat $f | lowercase.perl > $OUT
        cat $OUT | normalize-punctuation.perl -l $l > $OUT.norm
        cat $OUT.norm | tr -d '[:punct:]' > $OUT.norm.nopunct
        cat $OUT.norm.nopunct | tokenizer.perl -l $l -threads 8 > $OUT.norm.nopunct.tok.$l
      fi
    done
  done
done

# Cross product of pairs
for TYPE in train val; do
  if [ ! -f "${TYPE}_all.shuf.idxs" ]; then
    rm tmp.* &> /dev/null
    OUT="tmp.$TYPE"
    for sf in `ls --color=none $SL/$TYPE/*tok.${SL}`; do
      for tf in `ls --color=none $TL/$TYPE/*tok.${TL}`; do
        echo "Merging $sf and $tf"
        paste -d"|" $sf $tf | nl -v0 -w1 -s'|' >> $OUT
      done
    done

    # Shuffle
    echo "Shuffling"
    shuf < $OUT > ${OUT}.shuf

    # Split the file into image idxs, srcs and trgs
    echo "Splitting back into idxs, srcs, trgs"
    awk -F'|' '{print $1 >> "tmp.img"; print $2 >> "tmp.src"; print $3 >> "tmp.trg"}' ${OUT}.shuf

    # Final files of all combinations
    mv tmp.src tmp.test.$SL
    mv tmp.trg tmp.test.$TL

    # Clean out imbalanced pairs
    clean-corpus-n-frac.perl tmp.test $SL $TL tmp.test.clean 1 $MAX 3 tmp.lines.retained
    $SELECT_LINES tmp.img tmp.lines.retained > ${TYPE}_all.shuf.idxs

    mv tmp.test.clean.$SL ${TYPE}_all.shuf.tok.${SL}
    mv tmp.test.clean.$TL ${TYPE}_all.shuf.tok.${TL}
  fi
done

# Create training dictionary
if [ ! -f train_all.shuf.tok.${SL}.pkl ]; then
  # Create training vocabularies
  $BDIC train_all.shuf.tok.$SL train_all.shuf.tok.$TL
fi

###########################################################
# Create a sample of 1000 validation from each val split
# Use their german sides for multiple reference evaluation
N_SMALL=50
OUT="tmp.val_small"
rm tmp.* &> /dev/null
for sf in `ls --color=none $SL/val/*tok.${SL}`; do
  for tf in `ls --color=none $TL/val/*tok.${TL}`; do
    echo "Merging first $N_SMALL lines of $sf and $tf"
    head -n$N_SMALL $sf > tmp_val_small.$SL
    head -n$N_SMALL $tf > tmp_val_small.$TL
    paste -d"|" tmp_val_small.$SL tmp_val_small.$TL | nl -v0 -w1 -s'|' >> $OUT
  done
done

# Split the file into image idxs, srcs and trgs
echo "Splitting back into idxs, srcs, trgs"
awk -F'|' '{print $1 >> "tmp.img"; print $2 >> "tmp.src"; print $3 >> "tmp.trg"}' ${OUT}

# Final files of all combinations
mv tmp.src tmp.test.$SL
mv tmp.trg tmp.test.$TL

# Clean out imbalanced pairs
clean-corpus-n-frac.perl tmp.test $SL $TL tmp.test.clean 1 $MAX 3 tmp.lines.retained
$SELECT_LINES tmp.img tmp.lines.retained > val_small.shuf.idxs

mv tmp.test.clean.$SL val_small.shuf.tok.${SL}
mv tmp.test.clean.$TL val_small.shuf.tok.${TL}

rm tmp*
