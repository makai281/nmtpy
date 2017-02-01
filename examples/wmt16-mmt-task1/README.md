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

Run `scripts/02.bpe.sh ~/nmtpy/data/wmt16-task1` to learn a joint BPE model on both
sides with 20k merge operations and apply it to train, val and test files. The new BPE
processed files will have a `.bpe` in their names.

This script will also create `nmtpy` vocabulary files under the same output folder:
  - `train.norm.max50.tok.lc.bpe.en.pkl`: 8530 tokens 
  - `train.norm.max50.tok.lc.bpe.de.pkl`: 12763 tokens

### Train a Monomodal NMT

Run `nmt-train -c wmt16-task1-monomodal.conf -t` to train a monomodal NMT on this
corpus.

### Train a Multimodal NMT
