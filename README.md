# nmtpy

nmtpy is a suite of Python tools, primarily based on the starter code provided in [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial) for training neural machine translation networks using Theano. A non-exhaustive list of differences between nmtpy and dl4mt-tutorial is as follows:
  - 

## Requirements

You need the following libraries installed in order to use nmtpy:
  - numpy
  - Theano >= 0.8 (or a GIT checkout with `tensor.nnet.logsoftmax` available)
  - six

We recommend using Anaconda Python distribution which is equipped with Intel MKL (Math Kernel Library) greatly
improving CPU decoding speeds during beam search. With a correct compilation and installation, you should achieve
similar performance with OpenBLAS as well but the setup procedure may be difficult to follow for inexperienced ones.

**Note on MKL**: If you are using Anaconda, make sure to just pass `ldflags = -lmkl_rt` in `[blas]` section of your `.theanorc`.

nmtpy currently only supports Python 2.7 but we plan to move towards Python 3 in the future.

## Installation
After fulfilling the dependencies, create an easy-install link to the GIT repository so that whenever you issue a `git pull`, you start using the latest version automatically:

```
$ python setup.py develop
```

This is also crucial for `nmt-train` in case you create a new model file under `nmtpy/models`.

## Utilities

Once you do the installation correctly, you'll have access to 3 scripts in your Python environment:

- `nmt-train`: Main training loop
- `nmt-extract`: A small utility to save the source/target word embeddings of your networks. Note that you have to keep the layer names as `Wemb_enc` and `Wemb_dec` for this to work correctly.
- `nmt-translate`: The utility which does the beam-search and translation generation.
- `nmt-build-dict`: The utility to create vocabulary files (.pkl) from corpora.
- `nmt-coco-metrics`: Evaluates a translation file with multiple references using BLEU1-4, METEOR, CIDEr and ROUGE.
