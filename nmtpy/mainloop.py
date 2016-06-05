#!/usr/bin/env python
from math import ceil
from shutil import copy

class MainLoop(object):
    def __init__(self, model, logger, train_args):
        # This is the nmtpy model and the logger
        self.model          = model
        self.model_path     = train_args.model_path
        self.__log          = logger

        # Counters
        self.update_ctr     = 1
        self.epoch_ctr      = 1
        self.valid_ctr      = 0

        # Only used for SGD
        self.lrate          = train_args.lrate

        self.early_bad      = 0
        self.early_stop     = False
        self.proceed        = True
        self.save_iter      = train_args.save_iter
        self.max_updates    = train_args.max_iteration
        self.max_epochs     = train_args.max_epochs
        self.early_patience = train_args.patience
        self.valid_metric   = train_args.valid_metric
        self.do_beam_search = self.valid_metric != 'px'

        # Number of samples to produce
        self.n_samples      = 5

        # Losses and metrics
        self.__batch_losses         = []
        self.__valid_losses         = []
        self.__valid_metrics        = []
        # List of lists
        self.__train_losses         = []

        self.batch_size     = train_args.batch_size
        self.valid_freq     = train_args.valid_freq
        self.sample_freq    = train_args.sample_freq
        self.do_sampling    = self.sample_freq > 0

        # If valid_freq == 0, do validation at end of epochs
        if self.valid_freq == 0:
            self.valid_freq = ceil(self.model.train_iterator.n_samples / float(self.batch_size))

        # Periodic hooks to run during training
        self.__hooks        = []

    def __header(self, msg):
        """Pretty prints a message with ending dashes."""
        self.__log.info(msg)
        self.__log.info('-' * len(msg))

    def save_best_model(self):
        """Overwrites best on-disk model and saves it as a different file optionally."""
        self.__log.info('Saving the best model')
        self.early_bad = 0
        self.model.save_params(self.model_path,
                              #valid_losses=valid_losses, TODO
                              #metric_history=metric_history, TODO
                              uidx=self.update_ctr, **unzip(self.model.tparams))

        # Save each best model as different files
        # Can be useful for ensembling
        if self.save_iter:
            self.__log.info('Saving best model at iteration %d' % self.update_ctr)
            model_path_uidx = '%s.iter%d.npz' % (os.path.splitext(self.model_path)[0], self.update_ctr)
            copy(self.model_path, model_path_uidx)

    def __update_stop_conditions(self):
        if self.update_ctr == self.max_updates:
            self.proceed = False
        if self.epoch_ctr == self.max_epochs:
            self.proceed = False
        if self.early_bad == self.early_patience:
            self.proceed = False

    def __update_lrate(self):
        """Update learning rate by annealing it (for SGD)."""
        # TODO
        self.lrate = self.lrate

    def __run_epoch(self):
        self.__header('Starting Epoch %d' % self.epoch_ctr)
        batch_losses = []
        # Iterate over batches
        for batch_dict in self.model.train_iterator:
            self.update_ctr += 1
            self.model.set_dropout(True)

            # Forward pass and record batch loss
            loss = self.model.f_grad_shared(*batch_dict.values())
            batch_losses.append(loss)
            self.__update_lrate()

            # Backward pass
            self.model.f_update(self.lrate)

            # verbose
            if self.update_ctr % 10 == 0:
                self.__log.info("Epoch: %4d, update: %7d, Cost: %10.2f" % (self.epoch_ctr,
                                                                           self.update_ctr,
                                                                           loss))
            # Do sampling
            if self.do_sampling and self.update_ctr % self.sample_freq == 0:
                self.__do_sample(batch_dict)

            # Do validation
            if self.epoch_ctr >= self.valid_start and self.update_ctr % self.valid_freq == 0:
                self.__do_validation()

        # Save epoch losses
        self.__train_losses.append(batch_losses)
        mean_loss = np.array(batch_losses).mean()
        self.__log.info("Epoch %d finished with mean loss %.5f" % (self.epoch_ctr, mean_loss))

    def __do_sample(self, bdict):
        """Generates samples and prints them."""
        samples = self.model.generate_samples(bdict, self.n_samples)
        if samples is not None:
            for src, truth, sample in samples:
                if src:
                    self.__log.info("Source: %s" % src)
                self.__log.info("Sample: %s" % sample)
                self.__log.info(" Truth: %s" % truth)

    def __do_validation(self):
        # Compute validation loss
        self.model.set_dropout(False)
        self.__valid_losses.append(self.model.val_loss())
        self.__log.info("[Validation %2d] LOSS = %5.5f" % (len(self.__valid_losses), self.__valid_losses[-1]))
        if self.do_beam_search:
            results = model.run_beam_search(beam_size=12,
                                            n_jobs=self.njobs,
                                            metric=self.valid_metric,
                                            mode=self.decoder_mode)

            # We'll receive the requested metrics in a dict
            metric_results.append(results)
            for _, v in sorted(results.iteritems()):
                log.info("[Validation %2d] %s" % (len(metric_history)+1, v[0]))

            # Pick the one selected as valid_metric and add it to metric_history
            metric_history.append(results[args.valid_metric][1])


    def __print_eval_summary(self):
        if len(self.metric_history) > 0:
            best_metric_uidx = np.argmax(np.array(self.metric_history))
            best_metric = self.

    def run(self):
        while self.proceed:
            self.__run_epoch()

            # Update stopping conditions
            # FIXME: Not here actually but after each update? in eval?
            self.__update_stop_conditions()
