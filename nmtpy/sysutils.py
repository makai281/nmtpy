#!/usr/bin/env python
import os
import sys
import time
import select
import cPickle
import inspect
import tempfile
import subprocess

from . import cleanup

"""System related utility functions."""
def ensure_dirs(dirs):
    try:
        for d in dirs:
            os.makedirs(d)
    except OSError as oe:
        pass

def real_path(p):
    return os.path.abspath(os.path.expanduser(p))

def fix_model_options(d):
    """Removes old stuff to make old models work with latest code."""
    # Remove fault theano trng object from dict
    if "trng" in d:
        del d["trng"]

    data = d['data']
    # Remove iterator types from data dict
    for k, v in data.iteritems():
        if isinstance(v, list) and v[0] in ["img_feats", "text", "bitext"]:
            d['data'] = dict([[k, v[1]] for k,v in data.iteritems()])

    return d

def readable_size(n):
    sizes = ['K', 'M', 'G']
    fmt = ''
    size = n
    for i,s in enumerate(sizes):
        nn = n / (1000.**(i+1))
        if nn >= 1:
            size = nn
            fmt = sizes[i]
        else:
            break
    return '%.1f%s' % (size, fmt)

def get_temp_file(suffix="", name=None, delete=False):
    """Creates a temporary file under /tmp. If name is not None
    it will be used as the temporary file's name."""
    if name:
        name = os.path.join("/tmp", name)
        t = open(name, "w")
        cleanup.register_tmp_file(name)
    else:
        _suffix = "_nmtpy_%d" % os.getpid()
        if suffix != "":
            _suffix += suffix

        t = tempfile.NamedTemporaryFile(suffix=_suffix, delete=delete)
        cleanup.register_tmp_file(t.name)
    return t

def get_valid_evaluation(model_path, beam_size, n_jobs, metric, mode, pkl_path=None, out_file=None):
    cmd = ["nmt-translate", "-b", str(beam_size), "-D", mode,
           "-j", str(n_jobs), "-m", model_path, "-M", metric]
    if pkl_path:
        cmd.extend(["-p", pkl_path])
    if out_file:
        cmd.extend(["-o", out_file])
    # nmt-translate prints a dict of metrics
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stdout)
    cleanup.register_proc(p.pid)
    out, err = p.communicate()
    cleanup.unregister_proc(p.pid)
    return eval(out.splitlines()[-1].strip())

### GPU & PBS related functions
def create_gpu_lock(used_gpu):
    name = "gpu_lock.pid%d.gpu%s" % (os.getpid(), used_gpu)
    lockfile = get_temp_file(name=name)
    lockfile.write("[nmtpy] %s\n" % name)

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
        create_gpu_lock(int(which.replace("gpu", "")))
        return which
    # auto favors GPU in the first place
    elif which == 'auto':
        try:
            out = subprocess.check_output(["nvidia-smi", "-q"])
        except OSError as oe:
            # Binary not found, fallback to CPU
            return "cpu"

        # Find out about GPU usage
        usage = ["None" in l for l in out.split("\n") if "Processes" in l]
        try:
            # Get first unused one
            which = usage.index(True)
        except ValueError as ve:
            # No available GPU on this machine
            return "cpu"

        lock_file = create_gpu_lock(which)
        return ("gpu%d" % which)
