#!/bin/bash

LANG=$1
shift 1
TOKHYP=$1
shift 1
DETOKREF=$@

detokenizer.perl -l $LANG < $TOKHYP > ${TOKHYP}.detok
multi-bleu.perl $DETOKREF < ${TOKHYP}.detok
nmt-coco-metrics -l $LANG -p $TOKHYP $DETOKREF
