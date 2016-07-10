#!/usr/bin/env python
import sys

if __name__ == '__main__':
    try:
        n_samples = int(sys.argv[1])
        hyps_file = sys.argv[2]
    except:
        print "Usage: %s <# samples in the set> <hyps file>" % sys.argv[0]
        sys.exit(1)

    lines = []
    with open(hyps_file) as f:
        for line in f:
            lines.append(line.strip())

    n_splits = len(lines) / n_samples

    for i in range(1, n_splits+1):
        with open(hyps_file + '.%d' % i, 'w') as f:
            for line in lines[(i-1)*n_samples:i*n_samples]:
                f.write(line.split(" ||| ")[1] + '\n')
