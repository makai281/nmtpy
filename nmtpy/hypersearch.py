"""Hyper-parameter search."""

import numpy as np

from nmtpy.mainloop import MainLoop
from nmtpy.nmtutils import DotDict

# How many runs?
N_ITER      = 20
MAX_EPOCHS  = 10
GCLIP       = 1.0

def hypersearch(model, log, args):
    results = []

    l2_factors = 10**(-np.arange(5,9).astype('float64'))
    for l2 in l2_factors:
        # Set seed
        np.random.seed(1234)

        log.info("Loading data")
        model.load_data()
        log.info("Initializing parameters")
        model.init_params()
        model.init_shared_variables()
        log.info("Number of parameters: %s" % model.get_nb_params())
        log.info("Building model")
        cost = model.build()

        args.decay_c = l2
        # Get regularized training cost
        cost = model.get_regularized_cost(cost, args.decay_c, 0)

        # Build optimizer
        log.info("Building optimizer with regularized cost (l2: %.10f)" % l2)
        model.build_optimizer(cost, GCLIP, debug=False)

        # Enable dropout in training if any
        model.set_dropout(True)

        # Create mainloop
        loop = MainLoop(model, log, args)
        loop.beam_size = 12
        loop.save_best = False
        loop.f_verbose = 100
        loop.max_epochs = MAX_EPOCHS
        loop.run()
