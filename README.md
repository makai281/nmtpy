# nmtpy

nmtpy is a suite of Python tools, primarily based on the starter code provided in [dl4mt-material](https://github.com/kyunghyuncho/dl4mt-material) for training neural machine translation networks using Theano.

## Table of Contents

- [Installation](#installation)
- [Utilities](#utilities)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Configuration](#configuration)
- [Training](#training)
- [References](#references)

## Installation

You need the following libraries installed in order to use nmtpy:
  - numpy
  - numexpr (optional for fast logarithms)
  - theano >= 0.8 (or a GIT checkout with `tensor.nnet.logsoftmax` available)
  - six

After fulfilling the dependencies, create a easy-install link to the GIT repository so that whenever you issue a `git pull`, you start using the latest version automatically:

```
python setup.py develop
```

This is also crucial for `nmt-train` in case you create a new model file under `nmtpy/models`.

## Utilities

Once you do the installation correctly, you'll have access to 3 scripts in your Python environment:

- `nmt-train`: Main training loop
- `nmt-translate`: The utility which does the beam-search and translation generation.
- `nmt-build-dict`: The utility to create vocabulary files (.pkl) from corpora.

## Preprocessing

`nmtpy` does a minimum of preprocessing so it's up to you to create the correctly tokenized
versions of your corpora.

The only thing that you can modify dynamically during training is
the number of words in the vocabulary. The `nmt-build-dict` never limits the vocabulary, instead
it creates a frequency-ordered dictionary containing the full vocabulary. If you set `n-words-src`
and `n-words-trg` to a value different than zero, that value becomes the upper limit for the
corresponding vocabulary.

## Models

The only kind-of-stable model in the repository is the NMT with attention model [0] which is implemented
in `nmtpy/models/attention.py`.

We provide a set of implemented methods common to each model in `nmtpy/models/basemodel.py` to load, save
the model options and parameters, to enable/disable dropout, to initialize shared variables, etc. Feel
free to look at the `basemodel` for further details.

If you'd like to implement a completely new model, you can start by copying `attention.py` under a different
name, let's say `my_new_model.py` in the `nmtpy/models` folder.

Next step is to implement the following set of abstract methods for your model:
  - `load_data()`
  - `init_params()`
  - `build()`
  - `build_sampler()`

(Note that you can also redefine an already existing method if your model needs modifications for that
method.)

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

[0]: Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
