#!/usr/bin/env python
from math import ceil
from shutil import copy

# The serialization of this mainloop can ease
# resuming of trainings.

class MainLoop(object):
    def __init__(self, model, logger, train_args):
        # This is the nmtpy model and the logger
        self.model = model
        self.__log = logger
        self.__args = train_args

        # Counters
        self.update_ctr = 1
        self.epoch_ctr = 1
        self.valid_ctr = 0

        # Only used for SGD
        self.lrate = self.__args.lrate

        # Breaking conditions
        self.early_bad = 0
        self.early_stop = False
        self.proceed = True
        self.max_updates = self.__args.max_iteration
        self.max_epochs = self.__args.max_epochs
        self.early_patience = self.__args.patience

        # Losses and metrics
        self.__batch_losses = []
        self.__mean_epoch_losses = []

        self.batch_size = self.__args.batch_size
        self.valid_freq = self.__args.valid_freq
        # If valid_freq == 0, do validation at end of epochs
        if self.valid_freq == 0:
            self.valid_freq = ceil(self.model.train_iterator.n_samples / float(self.batch_size))

    def save_best_model(self):
        """Overwrites best on-disk model and saves it as a different file optionally."""
        self.__log.info('Saving the best model')
        self.early_bad = 0
        self.model.save_params(self.__args.model_path,
                              #valid_losses=valid_losses, TODO
                              #metric_history=metric_history, TODO
                              uidx=self.update_ctr, **unzip(self.model.tparams))

        # Save each best model as different files
        # Can be useful for ensembling
        if self.__args.save_iter:
            self.__log.info('Saving best model at iteration %d' % self.update_ctr)
            model_path_uidx = '%s.iter%d.npz' % (os.path.splitext(self.__args.model_path)[0], self.update_ctr)
            copy(self.__args.model_path, model_path_uidx)


    def __header(self, msg):
        """Prints a message with trailing dashes."""
        self.__log.info(msg)
        self.__log.info('-' * len(msg))

    def __update_stop_conditions(self):
        if self.update_ctr == self.max_updates:
            self.proceed = False
        if self.epoch_ctr == self.max_epochs:
            self.proceed = False
        if self.early_bad == self.early_patience:
            self.proceed = False

    def __update_lrate(self):
        """Update learning rate by annealing it (for SGD)."""
        self.lrate = self.lrate

    def __run_epoch(self):
        self.__header('Starting Epoch %d' % self.epoch_ctr)
        # Iterate over batches
        for batch_dict in self.model.train_iterator:
            self.update_ctr += 1
            self.model.set_dropout(True)

            # Forward pass and record batch loss
            batch_loss = self.model.f_grad_shared(*batch_dict.values())
            self.__batch_losses.append(batch_loss)
            self.__update_lrate()
            # Update weights. lrate only useful for SGD
            self.model.f_update(self.lrate)

            # verbose
            if self.update_ctr % 10 == 0:
                self.__log.info("Epoch: %4d, update: %7d, Cost: %10.2f" % (self.epoch_ctr,
                                                                           self.update_ctr,
                                                                           batch_loss))

            # Do sampling
            self.__do_sample()
            # Do validation
            self.__do_validation()

            # TODO: Index based periodic stuff can be further done
            # in a single check like pseudo timer

    def run(self):
        while self.proceed:
            self.__run_epoch()


            # Update stopping conditions
            # FIXME: Not here actually but after each update? in eval?
            self.__update_stop_conditions()
