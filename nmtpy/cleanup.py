import os
import sys
import signal
import atexit

temp_files = set()
subprocesses = set()

def register_tmp_file(f):
    temp_files.add(f)

def register_proc(pid):
    subprocesses.add(pid)

def unregister_proc(pid):
    subprocesses.remove(pid)

def __cleanup():
    # Correctly remove temporary files
    for f in temp_files:
        try:
            os.unlink(f)
        except:
            pass
        else:
            print "Cleaned up %s" % f

    # Send SIGTERM to subprocesses if any
    for p in subprocesses:
        try:
            os.kill(p, signal.SIGTERM)
        except:
            pass

    os._exit(0)

def register_handler():
    def handler(signum, frame):
        __cleanup()

    # Register SIGINT and SIGTERM
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    atexit.register(__cleanup)

