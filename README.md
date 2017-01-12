# nmtpy

**nmtpy** is a suite of Python tools, primarily based on the starter code provided in [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial) for training neural machine translation networks using Theano.

The basic motivation behind forking **dl4mt-tutorial** was to create a framework where it would be
easy to implement a new model by just copying and modifying an existing model class (or even
inheriting from it and overriding some of its methods).

To achieve this purpose, **nmtpy** tries to completely isolate training loop, beam search
and model definition:
  - `nmt-train` script to initiate a training experiment
  - `nmt-translate` to produce model-agnostic translations. You just pass a trained model's
  checkpoint file and it does its job.
  - A `BaseModel` and several other NMT models deriving from it which define the actual
  architecture.

A non-exhaustive list of differences between **nmtpy** and **dl4mt-tutorial** is as follows:

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
    - Simple shuffle
  - Improved parallel translation decoding on CPU
    - 620D/1000D NMT on 8 **Xeon E5-2690v2** using a beam size of 12: ~3400 words/sec
  - Export decoding informations into `json` for further visualization of attention coefficients
  
#### Deep Learning
  - Efficient SGD, Adadelta, RMSProp and ADAM
    - Single forward/backward theano function without intermediate variables
  - Ability to stop updating a set of weights by recompiling optimizer
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

nmtpy currently only supports Python 2.7 but we plan to move towards Python 3 in the future.

## Configuring Theano

Here is a basic `.theanorc` file recommended for best performance:

```
[global]
# Not so important as nmtpy will pick an available GPU
device = gpu0
# We use float32 everywhere
floatX = float32
# Keep theano compilation in RAM if you have a 7/24 available server
base_compiledir=/tmp/theano-%(user)s

[nvcc]
# This is already disabled by upstream Theano as well
fastmath = False

[cuda]
# CUDA 8.0 is better
root = /opt/cuda-7.5

[dnn]
# Make sure you use CuDNN as well
enabled = auto
library_path = /opt/CUDNN/cudnn-v5.1/lib64
include_path = /opt/CUDNN/cudnn-v5.1/include

[lib]
# Allocate 95% of GPU memory once
cnmem = 0.95
```

### Checking BLAS configuration

Recent Theano versions can automatically detect correct MKL flags. To check whether yours is working, run the following command:

```
$ python -c 'import theano; print theano.config.blas.ldflags'
-L/home/ozancag/miniconda/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -lm -Wl,-rpath,/home/ozancag/miniconda/lib
```

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
