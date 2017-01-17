# -*- coding: utf-8 -*-

DEFAULTS = {
        'weight_init':        'xavier',       # Can be a float for the scale of normal initializatio, "xavier" or "he".
        'batch_size':         32,             # Training batch size
        'optimizer':          'adam',         # adadelta, sgd, rmsprop, adam
        'lrate':              0.0004,         # Initial learning rate
        }

TRAIN_DEFAULTS = {
        'debug':              False,          # Dump graph and print theano node input/outputs
        'decay_c':            0.,             # L2 penalty factor
        'clip_c':             5.,             # Clip gradients above clip_c
        'alpha_c':            0.,             # Alpha regularization for attentional models (not quite tested)
        'seed':               1234,           # RNG seed
        'suffix':             "",             # Model suffix
        'save_iter':          False,          # Save each best valid weights to separate file
        'device_id':          'auto',         #
        'patience':           30,             # Early stopping patience
        'max_epochs':         200,            # Max number of epochs to train
        'max_iteration':      int(1e6),       # Max number of updates to train
        'valid_start':        1,              # Epoch which validation will start
        'valid_freq':         0,              # 0: End of epochs
        'valid_metric':       'bleu',         # bleu, px, meteor
        'valid_mode':         'single',       # Specific to WMTIterator (single/pairs/all)
        'sample_freq':        0,              # Sampling frequency during training (0: disabled)
        'njobs':              10,             # # of parallel CPU tasks to do beam-search
        'beam_size':          12,             # Allow changing beam size during validation
        }
