# -*- coding: utf-8 -*-
import os
import sys
import signal
import atexit

temp_files = set()
subprocesses = set()

def register_tmp_file(f):
    """Add new temp file to global set."""
    temp_files.add(f)

def register_proc(pid):
    """Add new process to global set."""
    subprocesses.add(pid)

def unregister_proc(pid):
    """Remove given PID from global set."""
    subprocesses.remove(pid)

def __cleanup():
    """Remove temporary files and kill leftover processes."""
    for f in temp_files:
        try:
            os.unlink(f)
        except:
            pass

    # Send SIGTERM to subprocesses if any
    for p in subprocesses:
        try:
            os.kill(p, signal.SIGTERM)
        except:
            pass

    os._exit(0)

def register_handler():
    """Setup cleanup() as SIGINT, SIGTERM and exit handler."""
    def handler(signum, frame):
        __cleanup()

    # Register SIGINT and SIGTERM
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    atexit.register(__cleanup)
