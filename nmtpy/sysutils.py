# -*- coding: utf-8 -*-
import os
import sys
import gzip
import tempfile
import subprocess

from . import cleanup

def ensure_dirs(dirs):
    """Create a list of directories if not exists."""
    try:
        for d in dirs:
            os.makedirs(d)
    except OSError as oe:
        pass

def real_path(p):
    """Expand UNIX tilde and return real path."""
    return os.path.realpath(os.path.expanduser(p))

def listify(l):
    """Encapsulate l with list[] if not."""
    return [l] if not isinstance(l, list) else l

def readable_size(n):
    """Return a readable size string."""
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
    """Creates a temporary file under /tmp."""
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

def get_valid_evaluation(model_path, beam_size, n_jobs, metric, mode, valid_mode='single'):
    """Run nmt-translate for validation during training."""
    cmd = ["nmt-translate", "-b", str(beam_size), "-D", mode,
           "-j", str(n_jobs), "-m", model_path, "-M", metric, "-v", valid_mode]

    # nmt-translate will print a dict of metrics
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stdout)
    cleanup.register_proc(p.pid)
    out, err = p.communicate()
    cleanup.unregister_proc(p.pid)
    results = eval(out.splitlines()[-1].strip())
    return results[metric]

def create_gpu_lock(used_gpu):
    """Create a lock file for GPU reservation."""
    name = "gpu_lock.pid%d.gpu%s" % (os.getpid(), used_gpu)
    lockfile = get_temp_file(name=name)
    lockfile.write("[nmtpy] %s\n" % name)

def fopen(filename, mode='r'):
    """GZIP-aware file opening function."""
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def find_executable(fname):
    """Find executable in PATH."""
    fname = os.path.expanduser(fname)
    if os.path.isabs(fname) and os.access(fname, os.X_OK):
        return fname
    for path in os.environ['PATH'].split(':'):
        fpath = os.path.join(path, fname)
        if os.access(fpath, os.X_OK):
            return fpath

def get_device(which='auto'):
    """Return Theano device to use by favoring GPUs first."""
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

def get_exp_identifier(args):
    """Return a representative string for the experiment."""

    names = [args.model_type]

    for k in sorted(args):
        if k.endswith("_dim"):
            # Only the first letter should suffice for now, e for emb, r for rnn
            names.append('%s%d' % (k[0], args[k]))

    name = '-'.join(names)

    # Append optimizer and learning rate
    name += '-%s_%.e' % (args.optimizer, float(args.lrate))

    # Append batch size
    name += '-bs%d' % args.batch_size

    # Validation stuff
    name += '-%s' % args.valid_metric

    if args.valid_freq > 0:
        name += "-each%d" % args.valid_freq
    else:
        name += "-eachepoch"

    if args.decay_c > 0:
        name += "-l2_%.e" % args.decay_c

    if 'emb_dropout' in args:
        name += "-do_%.1f_%.1f_%.1f" % (args.emb_dropout, args.ctx_dropout, args.out_dropout)

    if args.clip_c > 0:
        name += "-gc%d" % int(args.clip_c)

    if args.alpha_c > 0:
        name += "-alpha_%.e" % args.alpha_c

    if isinstance(args.weight_init, str):
        name += "-init_%s" % args.weight_init
    else:
        name += "-init_%.e" % args.weight_init

    # Append seed
    name += "-s%d" % args.seed

    if 'suffix' in args:
        name = "%s-%s" % (name, args.suffix)
        del args['suffix']

    return name

def get_next_runid(model_path, exp_name):
    # Log file, runs start from 1, incremented if exists
    i = 1

    while os.path.exists(os.path.join(model_path, "%s.%d.log" % (exp_name, i))):
        i += 1

    return i

def setup_train_args(args):
    # Get identifier name
    exp_name = get_exp_identifier(args)
    next_run_id = get_next_runid(args.model_path, exp_name)

    # Construct log file name
    log_fname = os.path.join(args.model_path, "%s.%d.log" % (exp_name, next_run_id))

    # Save new path
    args.model_path = os.path.join(args.model_path, "%s.%d" % (exp_name, next_run_id))

    return args, log_fname
