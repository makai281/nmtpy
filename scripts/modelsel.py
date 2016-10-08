#!/usr/bin/env python
import os
import time
import argparse
import subprocess
import xml.etree.ElementTree as ET

from collections import OrderedDict

from multiprocessing import Process, Queue
from threading import Semaphore

import uuid
import random
random.seed(1234)

# Will be initialized from inside main
SEMAPHORE_GPU = None

##################
# Helper functions
##################
def initial_spawn(config, params, defparams):
    """Spawn initial instances before going into parse loop."""
    processes = OrderedDict()
    while True:
        paramdict = sample_params(params, defparams)
        uid, ps = spawn_trainer(config, paramdict)
        if uid is None:
            break
        else:
            # Spawn successful
            print 'Launched trainer %s with PID %d' % (uid, ps.pid)
            processes[uid] = ps
            # It takes time for the child to take hold of the GPU
            time.sleep(3)
            if not SEMAPHORE_GPU.acquire(blocking=False):
                break

    return processes

def log_parser():
    pass

def sample_params(vals, defparams):
    """Sample given parameter pairs and extend with default params."""
    d = {}
    for param in vals:
        # Each param is formatted as:
        # param-name:one of l,i,f(min, max)
        key, value = param.split(':')
        vmin, vmax = eval(value[1:])
        val = random.uniform(vmin, vmax)
        if value[0] == 'l':
            d[key] = 10**val
        elif value[0] == 'i':
            d[key] = int(val)
        elif value[0] == 'f':
            d[key] = val

    d.update(defparams)
    return d

def get_empty_gpu():
    """Returns the first empty GPU on the machine as gpu<id> for passing to Theano."""
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-x', '-q']))
    for idx, gpu in enumerate(root.findall('gpu')):
        if gpu.find('processes').find('process_info') is None:
            return 'gpu%d' % idx
    return None

def spawn_trainer(conf, params):
    """Spawn GPU trainer and return the process instance."""
    gpuid = get_empty_gpu()
    if gpuid:
        cmd = ['nmt-train', '-D', gpuid, '-c', conf, '-e']

        for key, value in params.iteritems():
            cmd.append('%s:%s' % (key, value))

        print '[Spawning (%s)]' % " ".join(cmd)

        env = os.environ
        env['PYTHONUNBUFFERED'] = 'YES'
        return (uuid.uuid4(), subprocess.Popen(cmd, stderr=subprocess.PIPE, env=env))
    else:
        return None, None

def get_gpu_count():
    """Returns GPU count on the machine."""
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-x', '-q']))
    return len(root.findall('gpu'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='modelsel')
    parser.add_argument('-c', '--config'    , type=str, help='Base configuration file.', required=True)
    parser.add_argument('-m', '--max-epochs', type=int, help='Max epoch for each run.', default=20)
    parser.add_argument('-p', '--params'    , nargs='*', help='Hyper-parameters and their ranges: param-name:[lif](min,max). l:log, i:int, f:float.', required=True)

    args = parser.parse_args()

    # Get # of GPUs and create a semaphore
    gpu_count = get_gpu_count()
    SEMAPHORE_GPU = Semaphore(gpu_count)

    default_params = {'max-epochs'  : args.max_epochs,
                      'beam-size'   : 3,
                     }

    # Spawn first instances
    procs = initial_spawn(args.config, args.params, default_params)

    time.sleep(2)

    for p in procs.values():
        p.kill()
