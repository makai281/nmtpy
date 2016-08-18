from shutil import copy

import numpy as np

class MainLoop(object):
    def __init__(self, model, logger, train_args):
        # This is the nmtpy model and the logger
        self.model          = model
        self.model_path     = train_args.model_path
        self.__log          = logger

        # Counters
        self.uctr = 0   # update ctr
        self.ectr = 0   # epoch ctr
        self.vctr = 0   # validation ctr

        self.lrate          = train_args.lrate

        self.early_bad      = 0
        self.early_stop     = False
        self.save_iter      = train_args.save_iter
        self.max_updates    = train_args.max_iteration
        self.max_epochs     = train_args.max_epochs
        self.early_patience = train_args.patience
        self.valid_metric   = train_args.valid_metric
        self.do_beam_search = self.valid_metric != 'px'

        # Number of samples to produce
        self.n_samples      = 5

        # Losses and metrics
        self.epoch_losses         = []
        self.valid_losses         = []
        self.valid_metrics        = []

        self.batch_size     = train_args.batch_size
        self.f_valid        = train_args.f_valid
        self.f_sample       = train_args.f_sample
        self.do_sampling    = self.f_sample > 0
        # TODO: Do this an option as well
        self.f_verbose      = 10

        # TODO:
        # After validation metric stabilizes more or less
        # do frequent validations using perplexity and call
        # beam-search if PX is relatively better more than 10%
        self.dynamic_validation = train_args.dynamic_validation

        # If f_valid == 0, do validation at end of epochs
        self.epoch_valid    = (self.f_valid == 0)

    # TODO
    def save(self):
        """Serializes the loop to resume training."""
        # Save the last parameters as usual with:
        # 1. uctr, ectr, losses, metrics, etc.
        # 2. backward realted stuff like lrate? dunno
        model.save(args.model_path)

    # OK
    def print(self, msg, footer=False):
        """Pretty prints a message."""
        self.__log.info(msg)
        if footer:
            self.__log.info('-' * len(msg))

    def save_best_model(self):
        """Overwrites best on-disk model and saves it as a different file optionally."""
        self.print('Saving the best model')
        self.early_bad = 0
        model.save(args.model_path)

        # Save each best model as different files
        # Can be useful for ensembling
        if self.save_iter:
            self.print('Saving best model at iteration %d' % self.uctr)
            model_path_uidx = '%s.iter%d.npz' % (os.path.splitext(self.model_path)[0], self.uctr)
            copy(self.model_path, model_path_uidx)

    # TODO
    def __update_lrate(self):
        """Update learning rate by annealing it."""
        self.lrate = self.lrate
    
    # OK
    def print_loss(self, loss):
        if self.uctr % self.f_verbose == 0:
            self.print("Epoch: %4d, update: %7d, cost: %10.6f" % (self.ectr,
                                                                  self.uctr,
                                                                  loss))

    # OK
    def _train_epoch(self):
        """Represents a training epoch."""
        self.print('Starting Epoch %d' % self.ectr, True)

        batch_losses = []

        # Iterate over batches
        for data in self.model.train_iterator:
            self.uctr += 1
            self.model.set_dropout(True)

            # Forward/backward and get loss
            loss = self.model.train_batch(*data.values())
            batch_losses.append(loss)

            # verbose
            self.print_loss(loss)

            # Should we stop
            if self.uctr == self.max_updates:
                break

            # Update learning rate if requested
            self.__update_lrate()

            # Do sampling
            self.__do_sampling(data)

            # Do validation
            self.__do_validation()

        if self.uctr == self.max_updates:
            self.print("Max iteration %d reached." % self.uctr)
            return

        # An epoch is finished
        mean_loss = np.array(batch_losses).mean()
        self.epoch_losses.append(mean_loss)
        self.print("Epoch %d finished with mean loss %.5f" % (self.ectr, mean_loss))

    # OK
    def __do_sampling(self, data):
        """Generates samples and prints them."""
        if self.do_sampling and self.uctr % self.f_sample == 0:
            samples = self.model.generate_samples(data, self.n_samples)
            if samples is not None:
                for src, truth, sample in samples:
                    if src:
                        self.print("Source: %s" % src)
                    self.print.info("Sample: %s" % sample)
                    self.print.info(" Truth: %s" % truth)

    def __do_validation(self):
        # Compute validation loss
        if self.ectr >= self.valid_start and self.uctr % self.f_valid == 0:
            self.model.set_dropout(False)
            self.__valid_losses.append(self.model.val_loss())
            self.print("[Validation %2d] LOSS = %5.5f" % (len(self.__valid_losses), self.__valid_losses[-1]))
            if self.do_beam_search:
                results = model.run_beam_search(beam_size=self.valid_beam_size,
                                                n_jobs=self.njobs,
                                                metric=self.valid_metric,
                                                mode=self.decoder_mode)

                # We'll receive the requested metrics in a dict
                metric_results.append(results)
                for _, v in sorted(results.iteritems()):
                    log.info("[Validation %3d] %s" % (len(metric_history)+1, v[0]))

                # Pick the one selected as valid_metric and add it to metric_history
                metric_history.append(results[args.valid_metric][1])

    def dump_val_summary():
        if len(valid_losses) > 0:
            best_valid_idx = np.argmin(np.array(valid_losses))
            best_vloss = valid_losses[best_valid_idx]
            best_px = np.exp(best_vloss)
            log.info('[Validation %3d] Current Best Loss %5.5f (PX: %4.5f)' % (best_valid_idx + 1,
                                                                               best_vloss, best_px))

        if len(metric_history) > 0:
            best_metric_idx = np.argmax(np.array(metric_history))
            best_valid = metric_results[best_metric_idx]
            for _, v in sorted(best_valid.iteritems()):
                log.info("[Validation %3d] Current Best %s" % (best_metric_idx + 1, v[0]))

    def run(self):
        # We start 1st epoch
        while 1:
            self.ectr += 1
            self._train_epoch()

            # Should we stop?
            if self.ectr == self.max_epochs:
                break

            ##############
            # End of epoch
            ##############

        #################
        # End of training
        #################
