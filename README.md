# nmtpy

**nmtpy** is a suite of Python tools, primarily based on the starter code provided in [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial) for training neural machine translation networks using Theano. A non-exhaustive list of differences between **nmtpy** and **dl4mt-tutorial** is as follows:

#### General/API
  - No shell script, everything is in Python 
  - Overhaul object-oriented refactoring of the code
    - Clear separation of API and scripts that interface with the API
    - A `BaseModel` abstract class to derive from when you implement a new architecture
    - An `Iterator` abstract class to derive from for novel iterators
  - Simple text configuration files to define everything regarding a training experiment
  - Transparent cleanup mechanism to kill stale processes, remove temporary files
  - Simultaneous logging of training details to stdout and log file
  
#### Training/Inference
  - Plugin-like text filters for hypothesis post-processing (Example: BPE)
  - Early-stopping and checkpointing based on perplexity, BLEU or METEOR
    - `nmt-train` automatically calls `nmt-translate` during validation and returns the result back
    - Ability to add new metrics easily
  - Single `.npz` file to store everything about a training experiment
  - Improved numerical stability and reproducibility
  - Automatic free GPU selection using on `nvidia-smi`
  - Shuffling support between epochs:
    - [Homogeneous batches of same-length samples](https://github.com/kelvinxu/arctic-captions) to improve training speed
    - Simple permutation
  - Improved parallel translation decoding on CPU
    - 620D/1000D NMT on 8 **Xeon E5-2690v2** using a beam size of 12: ~3400 words/sec
  - Export decoding informations into `json` for further visualization of attention coefficients
  
#### Deep Learning
  - Efficient SGD, Adadelta, RMSProp and ADAM
    - Single forward/backward theano function without intermediate variables
  - Several recurrent blocks: GRU, Conditional GRU (CGRU) and LSTM
  - [Layer Normalization](https://github.com/ryankiros/layer-norm) support for GRU
  - Simple/Non-recurrent Dropout, L2 weight decay
  - Training and validation loss normalization for correct perplexity computation
  - [Tied target embeddings](https://arxiv.org/abs/1608.05859)
  - Glorot/Xavier, He, Orthogonal weight initializations

## Requirements

You need the following libraries installed in order to use nmtpy:
  - numpy
  - Theano >= 0.8 (or a GIT checkout with `tensor.nnet.logsoftmax` available)
  - six

We recommend using Anaconda Python distribution which is equipped with Intel MKL (Math Kernel Library) greatly
improving CPU decoding speeds during beam search. With a correct compilation and installation, you should achieve
similar performance with OpenBLAS as well but the setup procedure may be difficult to follow for inexperienced ones.

**Note on MKL**: If you are using Anaconda, make sure you pass `ldflags = -lmkl_rt` in `[blas]` section of your `.theanorc`.

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
