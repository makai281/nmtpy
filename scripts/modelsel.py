#!/usr/bin/env python
import os
import time
import argparse
import subprocess
import xml.etree.ElementTree as ET

from collections import OrderedDict
from threading import Semaphore

import uuid
import random
random.seed(1234)

# Will be initialized from inside main
INTERRUPTED = False
SEMAPHORE_GPU = None
PROCS = OrderedDict()
DEVNULL = open(os.devnull, 'w')

##################
# Helper functions
##################
def reap_processes():
    for k,v in PROCS.items():
        # Check alive
        if v.poll() is not None:
            del PROCS[k]
            SEMAPHORE_GPU.release()

def spawn_single(config, params, defparams):
    """Spawns a single instance."""
    if not SEMAPHORE_GPU.acquire(blocking=False):
        return False

    paramdict = sample_params(params, defparams)
    uid, ps = spawn_trainer(config, paramdict)
    if uid is None:
        SEMAPHORE_GPU.release()
        return False

    # Store process
    PROCS[uid] = ps

    print 'Spawned new worker PID=%d' % ps.pid
    return True

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

        env = os.environ
        env['PYTHONUNBUFFERED'] = 'YES'
        return (uuid.uuid4(), subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, env=env))
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
    print '# of GPUs available: %d' % gpu_count
    SEMAPHORE_GPU = Semaphore(gpu_count)

    default_params = {'max-epochs'  : args.max_epochs,
                      'beam-size'   : 3,
                     }

    # Spawn first instances
    def spawn_workers():
        if not INTERRUPTED:
            # Will spawn until all GPUs are used.
            while spawn_single(args.config, args.params, default_params):
                # It takes time for the child to take hold of the GPU
                time.sleep(5)

    while True:
        try:
            reap_processes()
            spawn_workers()
            if INTERRUPTED:
                break
            time.sleep(60*3)
        except KeyboardInterrupt as ke:
            print 'Will stop once current models are finished.'
            INTERRUPTED = True

    print 'Done.'
