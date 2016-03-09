#!/usr/bin/env python

import os
from string import digits
from .helpers import DotDict
from .sysutils import real_path

is_path = lambda p: p.startswith(('~', '/', '../', './'))
def check_get_path(p):
    if is_path(p):
        p = real_path(p)
        if not os.path.exists(p):
            raise Exception("%s doesn't exist." % p)
    return p

def parse_config_option(k, v):
    k = k.strip().replace("-", "_")
    v = v.strip()

    # list, tuple, dict
    if v.startswith(("[", "(", "{")):
        v = eval(v)
        if isinstance(v, list) or isinstance(v, tuple):
            v = [check_get_path(e) for e in v]
        elif isinstance(v, dict):
            for key, value in v.iteritems():
                if not isinstance(value, list):
                    v[key] = check_get_path(value)
                else:
                    v[key] = [check_get_path(e) for e in value]
    # Boolean
    elif v.lower().startswith(("false", "true")):
        v = eval(v.capitalize())
    # Path
    elif is_path(v):
        v = real_path(v)
    # Empty
    elif not v.strip('"\''):
        v = ""
    # Numbers
    elif v[0].startswith(tuple(digits)) or v[0] == '-' and v[1].startswith(tuple(digits)):
        v = float(v) if "." in v or "," in v else int(v)

    return {k: v}

class Config(object):
    def __init__(self, filename):
        # Load file content
        try:
            with open(filename, 'rb') as f:
                content = f.read().strip().split("\n")
        except Exception as e:
            raise(e)

        # Read and strip lines
        lines = [line.strip() for line in content]
        # Filter out empty and comment lines
        lines = [line for line in lines if line and line[0] != '#']

        buffered_k = None
        buffered_v = ""

        for line in lines:
            if line.endswith("\\"):
                # The data will continue
                if buffered_k is None:
                    buffered_k, buffered_v = line.split(":", 1)
                    buffered_v = buffered_v.rstrip("\\")
                else:
                    buffered_v += line.rstrip("\\")
            elif buffered_k:
                # A multiline option was processed. Handle it.
                buffered_v += line.rstrip("\\")
                k = buffered_k
                v = buffered_v.strip()
                buffered_k = None
            else:
                k,v = line.split(":", 1)

            self.__dict__.update(parse_config_option(k,v))

    def get(self):
        return DotDict(self.__dict__)

if __name__ == '__main__':
    import sys
    c = Config(sys.argv[1])
