Documentation
--

**nmtpy** is a suite of Python tools, primarily based on the starter code provided in [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial)
for training neural machine translation models using Theano.

The basic motivation behind forking **dl4mt-tutorial** was to create a framework where it would be
easy to implement a new model by just copying and modifying an existing model class (or even
inheriting from it and overriding some of its methods).

To achieve this purpose, **nmtpy** tries to completely isolate training loop, beam search,
iteration and model definition:
  - `nmt-train` script to initiate a training experiment
  - `nmt-translate` to produce model-agnostic translations. You just give it a model checkpoint file and it does its job.
  - An abstract `BaseModel` class to derive from to define your NMT architecture.
  - An abstract `Iterator` to derive from for your custom iterators.
