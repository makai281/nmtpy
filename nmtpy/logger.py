#!/usr/bin/env python

import logging

def singleton(cls):
    instances = {}
    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()

@singleton
class Logger(object):
    def __init__(self):
        self.formatter = logging.Formatter('%(asctime)s %(message)s')
        self._logger = logging.getLogger('Theano NMT')
        self._logger.setLevel(logging.DEBUG)
        self._ch = logging.StreamHandler()
        self._ch.setFormatter(self.formatter)
        self._logger.addHandler(self._ch)

    def set_file(self, log_file):
        self._fh = logging.FileHandler(log_file, mode='w')
        self._fh.setFormatter(self.formatter)
        self._logger.addHandler(self._fh)

    def get(self):
        return self._logger
