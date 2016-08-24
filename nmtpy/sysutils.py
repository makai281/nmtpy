#!/usr/bin/env python
import os
import sys
import gzip
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

def listify(l):
    if not isinstance(l, list):
        return [l]
    return l

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

def get_valid_evaluation(model_path, beam_size, n_jobs, metric, mode, out_file=None):
    cmd = ["nmt-translate", "-b", str(beam_size), "-D", mode,
           "-j", str(n_jobs), "-m", model_path, "-M", metric]
    if out_file:
        cmd.extend(["-o", out_file])
    # nmt-translate prints a dict of metrics
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stdout)
    cleanup.register_proc(p.pid)
    out, err = p.communicate()
    cleanup.unregister_proc(p.pid)
    results = eval(out.splitlines()[-1].strip())
    return results[metric]

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

def setup_train_args(args):
    # Check METEOR path
    if args.valid_metric == "meteor":
        if "meteor_path" in args:
            os.environ['METEOR_JAR'] = args['meteor_path']
        else:
            raise Exception("You need to provide 'meteor-path' in your configuration.")

    # Find out dimensional information
    dim_str = ""
    for k in sorted(args):
        if k.endswith("_dim"):
            dim_str += "%s_%d-" % (k, args[k])
    if len(dim_str) > 0:
        dim_str = dim_str[:-1]

    # Append learning rate
    args.lrate = float(args.lrate)
    opt_string = args.optimizer
    opt_string += "-lr_%.e" % args.lrate

    # Set model name
    name = "%s-%s-%s-bs_%d-valid_%s" % (args.model_type, dim_str, opt_string, args.batch_size, args.valid_metric)

    if args.valid_freq > 0:
        name += "-each_%d" % args.valid_freq
    else:
        name += "-each_epoch"

    if args.decay_c > 0:
        name += "-decay_%.e" % args.decay_c

    if args.clip_c > 0:
        name += "-gclip_%.1f" % args.clip_c

    if args.alpha_c > 0:
        name += "-alpha_%.e" % args.alpha_c

    if isinstance(args.weight_init, str):
        name += "-winit_%s" % args.weight_init
    else:
        name += "-winit_%.e" % args.weight_init

    if args.seed != 1234:
        name += "-seed_%d" % args.seed

    if len(args.get('suffix', '')) > 0:
        name = "%s-%s" % (name, args.suffix)

    if 'suffix' in args:
        del args['suffix']

    args.model_path = os.path.join(args.model_path, args.model_path_suffix)
    del args['model_path_suffix']

    ensure_dirs([args.model_path])

    # Log suffix
    logsuff = 'search' if args.hypersearch else 'log'

    # Log file
    i = 1
    log_file = os.path.join(args.model_path, "%s_run%d.%s" % (name, i, logsuff))

    while os.path.exists(log_file):
        i += 1
        log_file = os.path.join(args.model_path, "%s_run%d.%s" % (name, i, logsuff))

    # Save prefix
    args.model_path = os.path.join(args.model_path, "%s_run%d" % (name, i))

    return args, log_file
