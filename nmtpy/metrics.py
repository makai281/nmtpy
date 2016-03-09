#!/usr/bin/env python

import os
import re
from subprocess import Popen, PIPE, check_output
from tempfile import mkstemp

from functools import total_ordering

from .sysutils import find_executable

@total_ordering
class BLEUScore(object):
    def __init__(self, score=""):
        sc = re.findall("^BLEU = (.*), (.*)/(.*)/(.*)/(.*) \(BP=(.*), ratio=(.*), hyp_len=(.*), ref_len=(.*)\)$", score)
        self.__parsed = True
        if len(sc) > 0:
            sc = sc[0]
            self.bleu_score = float(sc[0])
            self.ngram_scores = [float(n) for n in sc[1:5]]
            self.brevity_penalty = float(sc[5])
            self.ratio = float(sc[6])
            self.hyp_len = int(sc[7])
            self.ref_len = int(sc[8])
        else:
            self.__parsed = False
            self.bleu_score = 0

    def __repr__(self):
        if self.__parsed == False:
            return "0"
        return "BLEU = %3.2f, %2.1f/%2.1f/%2.1f/%2.1f (BP=%.3f, ratio=%.3f, hyp_len=%d, ref_len=%d)" % \
                (self.bleu_score,
                 self.ngram_scores[0], self.ngram_scores[1], self.ngram_scores[2], self.ngram_scores[3],
                 self.brevity_penalty, self.ratio, self.hyp_len, self.ref_len)

    def __eq__(self, other):
        return self.bleu_score == other.bleu_score

    def __lt__(self, other):
        return self.bleu_score < other.bleu_score

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

    def score_files(self, ref_files, trans_file):
        cmdline = self.__cmdline[:]
        # Multiple reference files
        if isinstance(ref_files, list):
            for ref in ref_files:
                cmdline.append(ref)
        # Single reference file
        elif isinstance(ref_files, str):
            cmdline.append(ref_files)

        # Give the translation hypotheses as file object
        with open(trans_file, "rb") as ftrans:
            process = Popen(cmdline, stdout=PIPE, stderr=PIPE, stdin=ftrans)
            stdout, stderr = process.communicate()

        score = stdout.splitlines()
        if len(score) == 0:
            return BLEUScore()
        else:
            return BLEUScore(score[0].rstrip("\n"))

    def score_sentences(self, ref_sents, trans_sents):
        cmdline = self.__cmdline[:]
        temp_ref = isinstance(ref_sents, list)
        if temp_ref:
            (fh, fref) = mkstemp(suffix=".bleu")
            os.close(fh)
            with open(fref, "wb") as ref_file:
                ref_file.write("\n".join(ref_sents) + "\n")
        elif isinstance(ref_sents, str):
            # Filename
            fref = ref_sents

        cmdline.append(fref)

        # Hypotheses are sent through STDIN
        process = Popen(cmdline, stdout=PIPE, stderr=PIPE, stdin=PIPE)
        process.stdin.write("\n".join(trans_sents) + "\n")

        stdout, stderr = process.communicate()

        if temp_ref:
            os.unlink(fref)

        score = stdout.splitlines()
        if len(score) == 0:
            return BLEUScore()
        else:
            return BLEUScore(score[0].rstrip("\n"))

"""Meteor wrapper."""

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
