WMT16 Shared Task on Multimodal Translation
---

### Multi30k Translation Data

A copy of the original text files are available under `data/`. These files are downloaded
from [WMT16 Multimodal Task](http://www.statmt.org/wmt16/multimodal-task.html) webpage.

### Normalization and Tokenization

Make sure that the following scripts from the `mosesecoder` project are in your `$PATH`:
  - lowercase.perl
  - tokenizer.perl
  - normalize-punctuation.perl
  - clean-corpus-n-ratio.perl

Run `scripts/01-tokenize.sh ~/nmtpy/data/wmt16-task1` to:

  - Normalize punctuations
  - Tokenize
  - Filter out sentences with length **&lt; 2** and **&gt; 50** (only for training corpus)
  - Lowercase

train, val and test files from `data/` and save them under `~/nmtpy/data/wmt16-task1`.
**Note that** the output folder is in accordance with the configuration file
`wmt16-task1-monomodal.conf` so if you use another output folder, change the configuration
file as well.

### BPE Processing

Run `scripts/02-bpe.sh ~/nmtpy/data/wmt16-task1` to learn a joint BPE model on both
sides with 20k merge operations and apply it to train, val and test files. The new BPE
processed files will have a `.bpe` in their names.

This script will also create `nmtpy` vocabulary files under the same output folder:
  - `train.norm.max50.tok.lc.bpe.en.pkl`: 8530 tokens 
  - `train.norm.max50.tok.lc.bpe.de.pkl`: 12763 tokens

### Train a Monomodal NMT

Run `nmt-train -c wmt16-task1-monomodal.conf -t` to train a monomodal NMT on this
corpus. This small-sized monomodal NMT achieves state-of-the-art performance on this corpus.

```
When the training is over, you can translate the test set using the following command:
```
nmt-translate -m ~/nmtpy/models/wmt16-mmt-task1-monomodal/attention-e100-r100-adam_4e-04-bs32-meteor-each1000-l2_1e-05-do_0.2_0.4_0.4-gc5-init_xavier-s1235.1.npz \
              -S ~/nmtpy/data/wmt16-task1/test.norm.tok.lc.bpe.en \
              -o test.tok.de
```

This will produce a tokenized hypothesis file cleaned from BPE segmentations. Let's score this using `nmt-coco-metrics`:

```
nmt-coco-metrics -p -l de test.tok.de ~/nmtpy/data/wmt16-task1/test.norm.tok.lc.de
Language: de
The number of references is 1
Bleu_1: 0.67390 Bleu_2: 0.54240 Bleu_3: 0.44846 Bleu_4: 0.37333 CIDEr: 3.55837 METEOR: 0.57024 METEOR(norm): 0.57058 ROUGE_L: 0.67121
```

### Train a Multimodal NMT

Before training a multimodal system, we need to store the sentence pairs and
their relevant image IDs in pickled files so that `WMTIterator` can read them:

```
scripts/03-raw2pkl.py -i data/split_train.txt \
                      -l ~/nmtpy/data/wmt16-task1/train.lines \
                      -s ~/nmtpy/data/wmt16-task1/train.norm.max50.tok.lc.bpe.en \
                      -t ~/nmtpy/data/wmt16-task1/train.norm.max50.tok.lc.bpe.de \
                      -o ~/nmtpy/data/wmt16-task1/train.pkl

scripts/03-raw2pkl.py -i data/split_val.txt \
                      -s ~/nmtpy/data/wmt16-task1/val.norm.tok.lc.bpe.en \
                      -t ~/nmtpy/data/wmt16-task1/val.norm.tok.lc.bpe.de \
                      -o ~/nmtpy/data/wmt16-task1/val.pkl

scripts/03-raw2pkl.py -i data/split_test.txt \
                      -s ~/nmtpy/data/wmt16-task1/test.norm.tok.lc.bpe.en \
                      -t ~/nmtpy/data/wmt16-task1/test.norm.tok.lc.bpe.de \
                      -o ~/nmtpy/data/wmt16-task1/test.pkl
```
