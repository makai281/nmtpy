#!/usr/bin/env python
import os
import time
import socket
import select
import cPickle
import inspect
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

def start_translator(model_options, cmd=None):
    # This starts translate.py as a continuous
    # BLEU validation daemon to use during validation periods.
    if not cmd:
        frame = inspect.stack()[1]
        cmd = os.path.join(os.path.dirname(frame[1]), "translate.py")

    # This is to avoid AF_UNIX too long exception
    sock_name = sha1(model_options['model_path']).hexdigest()
    cmds = ["nmt-translate", "-m", model_options['valid_metric'], "daemon", "--socket-name", sock_name]
    p = subprocess.Popen(cmds, env=dict(os.environ, THEANO_FLAGS="device=cpu"))
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for i in range(10):
        try:
            sock.connect("\0" + sock_name)
        except socket.error as se:
            time.sleep(0.5)
        else:
            sock_send_data(sock, model_options)
            return p, sock

    return None, None

def sock_send_data(handle, d):
    c = cPickle.dumps(d, cPickle.HIGHEST_PROTOCOL)
    return handle.sendall(c + "%%%%%")

def sock_recv_data(handle):
    data = ""
    while 1:
        r = handle.recv(4096)
        data += r
        if r[-5:] == '%%%%%':
            break

    return cPickle.loads(data[:-5])

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
