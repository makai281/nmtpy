#!/bin/bash

PREFIX="https://raw.githubusercontent.com/cmu-mtlab/meteor/master/data/paraphrase"

for lang in de en fr; do
  if [ ! -f "paraphrase-${lang}.gz" ]; then
    echo "Downloading $lang paraphrase data..."
    curl "${PREFIX}-${lang}.gz" -o "paraphrase-${lang}.gz"
  fi
done
