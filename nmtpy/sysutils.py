#!/usr/bin/env python
import os
import time
import select
import cPickle
import inspect
import tempfile
import subprocess

from hashlib import sha1

"""System related utility functions."""
def ensure_dirs(dirs):
    try:
        for d in dirs:
            os.makedirs(d)
    except OSError as oe:
        pass

def real_path(p):
    return os.path.abspath(os.path.expanduser(p))

def get_valid_evaluation(model_path, beam_size=12):
    trans_fd, trans_fname = tempfile.mkstemp(suffix='.hyp')
    os.close(trans_fd)
    cmd = ["nmt-translate", "-b", str(beam_size),
           "-m", model_path, "-o", trans_fname]
    # let nmt-translate print a dict of metrics
    result = eval(subprocess.check_output(cmd).strip())
    os.unlink(trans_fname)
    return result

### GPU & PBS related functions
def create_gpu_lock(used_gpu):
    pid = os.getpid()
    lockfile = "/tmp/gpu_lock.pid%d.%s" % (pid, used_gpu)
    with open(lockfile, "w") as lf:
        lf.write("[Theano] Running PID %d on %s\n" % (pid, used_gpu))

    return lockfile

def remove_gpu_lock(lockfile):
    try:
        os.unlink(lockfile)
    except Exception as e:
        pass

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def find_executable(fname):
    fname = os.path.expanduser(fname)
    if os.path.isabs(fname) and os.access(fname, os.X_OK):
        return fname
    for path in os.environ['PATH'].split(':'):
        fpath = os.path.join(path, fname)
        if os.access(fpath, os.X_OK):
            return fpath

def get_gpu(which='auto'):
    if which == "cpu":
        return "cpu", None
    elif which.startswith("gpu"):
        # Don't care about usage. Some cards don't
        # provide that info in nvidia-smi as well.
        lock_file = create_gpu_lock(int(which.replace("gpu", "")))
        return which, lock_file
    # auto favors GPU in the first place
    elif which == 'auto':
        try:
            out = subprocess.check_output(["nvidia-smi", "-q"])
        except OSError as oe:
            # Binary not found, fallback to CPU
            return "cpu", None

        # Find out about GPU usage
        usage = ["None" in l for l in out.split("\n") if "Processes" in l]
        try:
            # Get first unused one
            which = usage.index(True)
        except ValueError as ve:
            # No available GPU on this machine
            return "cpu", None

        lock_file = create_gpu_lock(which)
        return ("gpu%d" % which), lock_file
