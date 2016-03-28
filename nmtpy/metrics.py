#!/usr/bin/env python

import os
import re
from subprocess import Popen, PIPE, check_output
from functools import total_ordering

from .sysutils import find_executable, real_path, get_temp_file

@total_ordering
class METEORScore(object):
    def __init__(self, score=""):
        if score:
            self.score = float(score)
        else:
            self.score = 0

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return "METEOR = %3.3f" % self.score

@total_ordering
class BLEUScore(object):
    def __init__(self, score=""):
        sc = re.findall("^BLEU = (.*), (.*)/(.*)/(.*)/(.*) \(BP=(.*), ratio=(.*), hyp_len=(.*), ref_len=(.*)\)$", score)
        self.__parsed = True
        if len(sc) > 0:
            sc = sc[0]
            self.score = float(sc[0])
            self.ngram_scores = [float(n) for n in sc[1:5]]
            self.brevity_penalty = float(sc[5])
            self.ratio = float(sc[6])
            self.hyp_len = int(sc[7])
            self.ref_len = int(sc[8])
        else:
            self.__parsed = False
            self.score = 0

    def __repr__(self):
        if self.__parsed == False:
            return "0"
        return "BLEU = %3.2f, %2.1f/%2.1f/%2.1f/%2.1f (BP=%.3f, ratio=%.3f, hyp_len=%d, ref_len=%d)" % \
                (self.score,
                 self.ngram_scores[0], self.ngram_scores[1], self.ngram_scores[2], self.ngram_scores[3],
                 self.brevity_penalty, self.ratio, self.hyp_len, self.ref_len)

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

"""MultiBleuScorer class."""
class MultiBleuScorer(object):
    def __init__(self, lowercase=False, script="multi-bleu.perl"):
        # For multi-bleu.script we give the reference(s) files as argv,
        # while the candidate translations are read from stdin.
        self.script = find_executable(script)
        self.lowercase = lowercase
        if not self.script:
            raise Exception("BLEU script %s not found." % self.script)

        self.__cmdline = [self.script]
        if self.lowercase:
            self.__cmdline.append("-lc")

    def compute(self, refs, hyps):
        cmdline = self.__cmdline[:]

        # Make reference files a list
        refs = [refs] if isinstance(refs, str) else refs
        cmdline.extend(refs)

        if isinstance(hyps, list):
            # Hypotheses are sent through STDIN
            process = Popen(cmdline, stdout=PIPE, stderr=PIPE, stdin=PIPE)
            process.stdin.write("\n".join(hyps) + "\n")
        elif isinstance(hyps, str):
            # Hypotheses is file
            with open(hyps, "rb") as fhyp:
                process = Popen(cmdline, stdout=PIPE, stderr=PIPE, stdin=fhyp)

        stdout, stderr = process.communicate()

        score = stdout.splitlines()
        if len(score) == 0:
            return BLEUScore()
        else:
            return BLEUScore(score[0].rstrip("\n"))

"""Meteor wrapper."""
class METEORScorer(object):
    def __init__(self):
        self.path = real_path(os.environ['METEOR_JAR'])
        if not os.path.exists(self.path):
            raise Exception("METEOR jar file not found.")

        self.__cmdline = ["java", "-Xmx2G", "-jar", self.path]

    def compute(self, refs, hyps, language="auto", norm=True):
        cmdline = self.__cmdline[:]

        if isinstance(hyps, list):
            # Create a temporary file
            with get_temp_file(suffix=".hyps") as tmpf:
                for hyp in hyps:
                    tmpf.write("%s\n" % hyp)

                cmdline.append(tmpf.name)

        elif isinstance(hyps, str):
            cmdline.append(hyps)

        # Make reference files a list
        refs = [refs] if isinstance(refs, str) else refs
        n_refs = len(refs)
        if n_refs > 1:
            # Multiple references
            # FIXME: METEOR can consume everything from stdin
            tmpff = get_temp_file(suffix=".refs")
            fname = tmpff.name
            tmpff.close()
            os.system('paste -d"\\n" %s > %s' % (" ".join(refs), fname))
            cmdline.append(fname)
        else:
            cmdline.append(refs[0])

        if language == "auto":
            # Take the extension of the 1st reference file, e.g. ".de"
            language = os.path.splitext(refs[0])[-1][1:]

        cmdline.extend(["-l", language])
        if norm:
            cmdline.append("-norm")

        if n_refs > 1:
            # Multiple references
            cmdline.extend(["-r", str(n_refs)])

        output = check_output(cmdline)

        score = output.splitlines()
        if len(score) == 0:
            return METEORScore()
        else:
            # Final score:              0.320320320320
            return METEORScore(score[-1].split(":")[-1].strip())

#######################################
SCORERS = {
            'meteor': METEORScorer,
            'bleu'  : MultiBleuScorer,
          }

def get_scorer(scorer):
    return SCORERS[scorer]

def get_all_scorers():
    return SCORERS
