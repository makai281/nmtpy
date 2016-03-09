#!/bin/bash

TRANSLATE=../translate.py
MODEL_ROOT=/lium/trad4a/wmt/2016/caglayan/theano-attention/new_models
LANG="de-en"
MODEL_PATH="$MODEL_ROOT/$LANG"
NBEST_PATH="$MODEL_PATH/nbest"

# Different beam sizes
BEAMS=(2 5 12 20 50)

mkdir -p $NBEST_PATH &> /dev/null


# Evaluate for each beam size
for (( i=0;  i < ${#BEAMS[@]}; i++ )); do
  for M in $(ls $MODEL_PATH/*npz); do
    BEAMSIZE=${BEAMS[$i]}
    MODEL_ID=`basename ${M/.npz/}`
    OUT="$NBEST_PATH/$MODEL_ID.beam$BEAMSIZE"
    OUT1BEST="$OUT.1best"

    if [ ! -f $OUT1BEST ]; then
      echo "Evaluating ${M}: beam-size=$BEAMSIZE"
      # Translate on validation set of the model, save the 1best
      # translations into $OUT1BEST
      python $TRANSLATE -b $BEAMSIZE model -m "$M" -o "$OUT1BEST"
    fi
  done
done




#######################################
# Find out best model based on BLEU
cd $NBEST_PATH
rm best_model.* &> /dev/null
ALL_MODELS=`grep "BLEU =" *bleu | sed 's/,/ /g' | sort -t" " -k 3.1bn`
# model.beamBS.1best.bleu
BEST_MODEL=`grep "BLEU =" *bleu | sed 's/,/ /g' | sort -t" " -k 3.1bn | tail -n1 | awk -F ':' '{print $1}'`
# model.beamBS
BEST_MODEL_BEAM=${BEST_MODEL/.1best.*/}
# model
BEST_MODEL_ID=${BEST_MODEL/.beam*/}

cat /dev/null > bleu_report.txt
echo "BLEU report"
echo "-----------"
echo "$ALL_MODELS" | while read -r x; do
  MODEL=`echo $x | awk -F ':' '{print $1}'`
  BLEU=`cat $MODEL`
  MODEL_BEAM_ID=${MODEL/.1best.*/}
  printf "%-120s %20s\n" $MODEL_BEAM_ID "$BLEU" | tee -a bleu_report.txt
done
echo "-----------"

# Setup links
ln -sf ../"$BEST_MODEL_ID.npz"     best_model.npz
ln -sf ../"$BEST_MODEL_ID.npz.pkl" best_model.npz.pkl
ln -sf ../"$BEST_MODEL_ID.log"     best_model.log

ln -sf "$BEST_MODEL"            best_model.1best.bleu
ln -sf "$BEST_MODEL_BEAM.1best" best_model.1best
