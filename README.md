# nmtpy

nmtpy is a suite of Python tools, primarily based on the starter code provided in [dl4mt-material](https://github.com/kyunghyuncho/dl4mt-material) for training neural machine translation networks using Theano.

## Table of Contents

- [News](#news)
- [Installation](#installation)
- [Utilities](#utilities)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Configuration](#configuration)
- [Training](#training)
- [References](#references)

## News

### v0.1 -> v0.2

A lot has been changed in this version. Notably:

 - A cleanup module is now used to handle temporary files so that no trace is left on the system even when you interrupt the training. The same module is also used to ensure correct interruption without seeing ugly exceptions in the terminal.
 - General code refactoring stuff. You can always look at the attention model to understand how your new model should look like.
 - A reference implementation for Google's show and tell paper [2]. This model uses the newly added LSTM layer from Kelvin Xu's arctic-captions code.
 - `sort` and `shuffle` parameters were dropped from configuration files. You can decide to shuffle your data inside your model's `load_data()` method.
 - `nmt-train` now respects your choice of early stopping metric, e.g. if you write `bleu` in your configuration file it won't try to compute meteor.
 - All layers now accepts a `scale` parameter which can be a float like `0.01` (which is the default) defining the standard deviation of the initial weights,
   `'xavier'` for using Xavier Glorot initialization scheme [3] or `'he'` for He' initialization scheme (mostly useful with ReLU and ConvNets) [4]. You can define
   your choice in the configuration file with `weight_init: xavier or he or a float like 0.01`.
 - New utilities: `nmt-coco-metrics` and `nmt-extract` which are detailed below in the Utilities section.
 - You can now give multiple reference files as a list in your configuration files or even as a wildcard. Look at `confs/attention-wmt16-task2-max50-ratio3-en-de.conf`
 - `valid_start: n` can be used in the configuration to start validation for early stopping in the n'th epoch. This may be useful to let the system train for at least n-1 epochs before doing validation.
 - **Important**: `nmt-translate` now uses your model's `load_valid_data()` method for being able to load the validation data exactly the same way as it's done during training. This dropped the necessity to implement model-aware iterator creation in nmt-translate which was cluttering the code. So take a look at the `attention` model for how it's done.

## Installation

You need the following libraries installed in order to use nmtpy:
  - numpy
  - theano >= 0.8 (or a GIT checkout with `tensor.nnet.logsoftmax` available)
  - six

After fulfilling the dependencies, create a easy-install link to the GIT repository so that whenever you issue a `git pull`, you start using the latest version automatically:

```
$ python setup.py develop
```

This is also crucial for `nmt-train` in case you create a new model file under `nmtpy/models`.

### Meteor paraphrases

The library now includes the official COCO evaluation tools of Microsoft which is used
to compute BLEU1-4, METEOR, CIDEr and ROUGE for captioning tasks. The folder `pycocoevalcap`
contains all the necessary tools for this except the paraphrase files for METEOR which
are too big to put inside a GIT repository. If you'd like to use the new `nmt-coco-metrics`
script to evaluate your translations/captions, you need to run the `download.sh` script
inside `pycocoevalcap/meteor/data` to fetch the paraphrase files.

Finally for correctly using METEOR, you have to export `METEOR_JAR` environment variable
to point to the `meteor-1.5.jar` file on your filesystem.

## Utilities

Once you do the installation correctly, you'll have access to 3 scripts in your Python environment:

- `nmt-train`: Main training loop
- `nmt-extract`: A small utility to save the source/target word embeddings of your networks. Note that you have to keep the layer names as `Wemb_enc` and `Wemb_dec` for this to work correctly.
- `nmt-translate`: The utility which does the beam-search and translation generation.
- `nmt-build-dict`: The utility to create vocabulary files (.pkl) from corpora.
- `nmt-coco-metrics`: Evaluates a translation file with multiple references using BLEU1-4, METEOR, CIDEr and ROUGE.

## Preprocessing

`nmtpy` does a minimum of preprocessing so it's up to you to create the correctly tokenized
versions of your corpora.

The only thing that you can modify dynamically during training is
the number of words in the vocabulary. The `nmt-build-dict` never limits the vocabulary, instead
it creates a frequency-ordered dictionary containing the full vocabulary. If you set `n-words-src`
and `n-words-trg` to a value different than zero, that value becomes the upper limit for the
corresponding vocabulary.

## Iterators

There are several iterators under `nmtpy/iterators` which are evolving in a fast pace so don't
consider them stable. After experimenting with the code for a while, I realized that **the best**
way to organize your data is to create a .pkl (pickle) file and to implement an iterator specific
to your data & model.

- `TextIterator`: This is for monolingual corpus. The data provided by this iterator can be set by the parameter `data_name` which is by default `x`. This means that the theano variable that you
build to access real data have to be named `x` as well.

- `BiTextIterator`: Same as above but for parallel corpora. Source side is by default `x` and target side is `y`. The relevant masks are called `x_mask` and `y_mask`.

- `FlickrIterator`: This is for reading Karpathy's deepsent `pkl` file for image captioning.

- `WMTIterator`: This is currently being developed for being able to read `pkl` files containing lists of samples in the following form: `[src, trg, img_id, img_name]`. We would be able to put `None`'s for unprovided modalities so this iterator will be able to work both for task1 and task2 of WMT16.

`MultiIterator` and `ImageFeatsIterator` are obsolete and are not meant to be used. They will be removed in a future version.

## Models

We provide a set of implemented methods common to each model in `nmtpy/models/basemodel.py` to load, save
the model options and parameters, to enable/disable dropout, to initialize shared variables, etc. Feel
free to look at the `basemodel` for further details.

If you'd like to implement a completely new model, you can start by copying the NMT with attention [1] implementation
`attention.py` under a different name, let's say `my_new_model.py` into the `nmtpy/models` folder.

Next step is to reimplement the methods for your model:
  - `load_data()`
  - `init_params()`
  - `build()`
  - `build_sampler()`
  - `beam_search()`

Note that you can also redefine an already existing method in `basemodel` if your model needs modifications
for that method. This happened to me for `gen_sample()` for example for some paper implementations.

### Model list

  - NMT with attention [1]: `nmtpy/models/attention.py`
  - Show and Tell [2]: `nmtpy/models/googlenic.py`

## Configuration

Create a new configuration file which will define all of the details for your experiment. The default values
for some of the training parameters are defined in the beginning of the `nmt-train` script:
  - `DEFAULTS` defines the default parameter values that will be passed to your model's `__init__()` method
  - `TRAIN_DEFAULTS` defines the default training values that will only be used by `nmt-train` script itself.

You can override all these parameters through the command-line parameters of `nmt-train` as well. Note that
the parameter keys in the configuration files are delimited with a `-` character which are then converted
to underscore during configuration parsing. So for example, `decay-c` in the configuration file will be
referred as `args.decay_c` in `nmt-train`.

Since everything is somehow managed automatically in terms of parameters, if your new requires a new
parameter, all you need to do is to add it to your configuration file. Since this is a new parameter
not known to the `nmt-train`, it will be automatically passed to the `__init__()` method of your model.

Finally, the `model-type` parameter in the configuration defines which model under `nmtpy/models` will
be used during this experiment.

## Training

Once your model and your configuration file is ready, you can launch the training with the following command:
```
$ nmt-train -c confs/my-model.conf
```

This will create a subfolder called `my-model` under the `model-path` directory defined in the configuration.
The model name will include a few important model parameter values in the name so that we don't override
a model accidentally during consecutive trials:

```
   model file: attention-embedding_dim_620-rnn_dim_1000-adadelta-bs_32-valid_bleu-decay_0.00050.npz
model options: attention-embedding_dim_620-rnn_dim_1000-adadelta-bs_32-valid_bleu-decay_0.00050.npz.pkl
 training log: attention-embedding_dim_620-rnn_dim_1000-adadelta-bs_32-valid_bleu-decay_0.00050.log
```

If you'd like to change a parameter that is not reflected to the model name, you can give a suffix
so that it's appended to the model name:

```
$ nmt-train -c confs/my-model.conf -s "second-run"
```

Now the model name will be set as `attention-embedding_dim_620-rnn_dim_1000-adadelta-bs_32-valid_bleu-decay_0.00050-second-run`

## References

  1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
  2. Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). Show and tell: A neural image caption generator. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 3156-3164).
  3. Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks.", International conference on artificial intelligence and statistics. 2010.
  4. Kaiming He et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. arXiv preprint arXiv:1502.01852.
