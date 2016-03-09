#!/usr/bin/env python

import os
import sys
import subprocess


class MeteorScorer(object):
    def __init__(self, path="/lium/buster1/caglayan/git/meteor/meteor-1.5.jar"):

        self.path = path
        self.__cmdline = ["java", "-Xmx2G", "-jar", self.path]

    def score_file(self, ref_file, trans_file, language="auto", norm=True):
        cmdline = self.__cmdline[:]
        cmdline.append(trans_file)
        cmdline.append(ref_file)
        if language == "auto":
            # Take the extension of the reference file, e.g. ".de"
            language = os.path.splitext(ref_file)[-1][1:]
        cmdline.extend(["-l", language])
        if norm:
            cmdline.append("-norm")

        output = subprocess.check_output(cmdline)

        score = output.splitlines()
        if len(score) == 0:
            return 0.
        else:
            # Final score:              0.320320320320
            return float(score[-1].split(":")[-1].strip())

if __name__ == '__main__':
    m = MeteorScorer()

    try:
        trans_file = sys.argv[1]
        ref_file = sys.argv[2]
    except IndexError as ie:
        print "Usage: %s <trans_file> <ref_file>" % sys.argv[0]
        sys.exit(1)

    print m.score_file(ref_file, trans_file)
