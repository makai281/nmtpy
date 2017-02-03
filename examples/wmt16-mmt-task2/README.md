# WMT16 Shared task on Multimodal Translation
## Task 2 - Cross-lingual Image Description Generation

### Multi30k Dataset

A copy of the original text files for Task 2 are available under `data/`. These files are downloaded
from [WMT16 Multimodal Task](http://www.statmt.org/wmt16/multimodal-task.html) webpage.
 
### Normalization and Tokenization

Make sure that the following scripts from the `mosesdecoder` project are in your `$PATH`:
  - tokenizer.perl
  - normalize-punctuation.perl

Run `scripts/01-tokenize.sh ~/nmtpy/data/wmt16-task2` to:
  - Normalize punctuations
  - Tokenize

train, val and test files from `data/` and save them under `~/nmtpy/data/wmt16-task2`.
**Note that** the output folder is in accordance with the configuration file
`wmt16-task2-monomodal.conf` so if you use another output folder, change the configuration
file as well.

### Preparing Data

`scripts/02-prepare.py` is a Python script that consumes all the tokenized data produces:
  - processed text files
  - `pkl` files to be used by `WMTIterator`
  - `nmtpy` dictionary files (`.pkl`) for source and target vocabularies
  
You can run the following command to prepare above files:
```
scripts/02-prepare.py -i data/split_all.txt \
                -t data/tok/train.*.en  -T data/tok/train.*.de \
                -v data/tok/val.*.en    -V data/tok/val.*.de \
                -e data/tok/test.*.en   -E data/tok/test.*.de \
                -l -s -d 5 -o ~/nmtpy/data/wmt16-task2 # lowercase, strippunct, minwordoccurrence5
```
