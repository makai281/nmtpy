#!/bin/bash

# number of mergeops
OUT=$1

if [ -z $OUT ]; then
  echo "Usage: $0 <output-dir>"
  exit 1
fi

MOPS=20000

BPEFILE="${OUT}/jointbpe${MOPS}.model"

echo "Training joint BPE with $MOPS merge ops"
cat "${OUT}/train.norm.max50.tok.lc".{en,de} | nmt-bpe-learn -s $MOPS > $BPEFILE

# Apply to train
echo "Applying BPE to train"
nmt-bpe-apply -c $BPEFILE -i "${OUT}/train.norm.max50.tok.lc.en" -o "${OUT}/train.norm.max50.tok.lc.bpe${MOPS}.en"
nmt-bpe-apply -c $BPEFILE -i "${OUT}/train.norm.max50.tok.lc.de" -o "${OUT}/train.norm.max50.tok.lc.bpe${MOPS}.de"

# Apply to val and test
for TYPE in "val" "test"; do
  echo "Applying BPE to $TYPE"
  nmt-bpe-apply -c $BPEFILE -i "${OUT}/${TYPE}.norm.tok.lc.en" -o "${OUT}/${TYPE}.norm.tok.lc.bpe${MOPS}.en"
  nmt-bpe-apply -c $BPEFILE -i "${OUT}/${TYPE}.norm.tok.lc.de" -o "${OUT}/${TYPE}.norm.tok.lc.bpe${MOPS}.de"
done

# Create dictionaries
nmt-build-dict "${OUT}/train.norm.max50.tok.lc.bpe$MOPS".{en,de} -o $OUT
