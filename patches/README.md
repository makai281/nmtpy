Stochastic behaviour of Theano
----

This folder contains 2 patches to remedy non-reproducible results
in Theano.

The first one (00) disables the use of a GPU op that is non-deterministic.
See [upstream issue](https://github.com/Theano/Theano/issues/3029) for more details.

The second one (01) is a fix for the scan operation that is already [included](https://github.com/Theano/Theano/commit/8769382ff661aab15dda474a4c74456037f73cc6)
in the `master` branch of Theano as of 17 August 2016. If your Theano copy is newer than that point, you don't
need to apply it.
