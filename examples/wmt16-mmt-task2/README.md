WMT16 Shared task on Multimodal Translation - Task 2
---

 - `data/`: This folder contains Multi30k dataset of the competition.
 - `scripts/`: This folder contains the necessary preprocessing and preparation scripts for nmtpy.
 
### Getting Image Features

In order to train the networks, you will need the convolutional features extracted from ResNet-50. You can download them from the links below:
 
 - [flickr30k_ResNets50_blck4_train.npy.xz](http://)
 - [flickr30k_ResNets50_blck4_val.npy.xz](http://)
 - [flickr30k_ResNets50_blck4_test.npy.xz](http://)
 
 After downloading the files under `data/`, extract them using the following command:
 ```
 $ xz -d <downloaded xz file>
 ```

### Normalization and Tokenization

`scripts/01-tokenize.sh` normalizes and tokenizes all the raw data under `data/` and saves the processed files.
(**Note:** You need to have `normalize-punctuation.perl` and `tokenizer.perl` from `moses` in your `$PATH`)
```
# Where are we?
$ pwd
/home/ozancag/git/nmtpy/examples/wmt16-mmt-task2

$ scripts/01-tokenize.sh data/tok
Normalizing punctuation and tokenizing data/test.1.en
Tokenizer Version 1.1
Language: en
Number of threads: 8
Normalizing punctuation and tokenizing data/test.2.en
Tokenizer Version 1.1
Language: en
Number of threads: 8
...
```

### Preparing Data

`scripts/02-prepare.py` is a Python script that consumes all the tokenized data from `data/tok` and produces:
  - processed text files
  - `pkl` files to be used by `WMTIterator`
  - `nmtpy` dictionary files (`pkl` as well) for source and target vocabularies
  
You can run the following snippet to obtain the above files:
```
OUTDIR="data/minvocab5.lc.nopunct.en-de"
mkdir -p $OUTDIR
scripts/02-prepare.py -i data/split_all.txt \
                -t data/tok/train.*.en  -T data/tok/train.*.de \
                -v data/tok/val.*.en    -V data/tok/val.*.de \
                -e data/tok/test.*.en   -E data/tok/test.*.de \
                -l -s -d 5 -o $OUTDIR # lowercase, strippunct, minwordoccurrence5
```
