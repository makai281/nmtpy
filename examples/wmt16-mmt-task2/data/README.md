Multi30k dataset
---

This is a reorganized folder containing extracted and renamed files from
the original WMT16 Multimodal Translation Task 2 train/dev/test splits, namely
the Multi30k dataset.

Original files can be downloaded from [here](http://www.statmt.org/wmt16/multimodal-task.html)

The files are organized as follows:
  - `train.[1-5].{en,de}`: 5 splits of training set each having 29K sentences
  - `val.[1-5].{en,de}`: 5 splits of dev set each having 1014 sentences
  - `test.[1-5].{en,de}`: 5 splits of test set each having 1000 sentences
  - `split_*.txt`: Text files containing sentence to image name mapping for each sets.

The patch file `fix-corpus-bugs.patch` is applied on top of these files to fix some weird noise in the dataset.
