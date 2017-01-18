# -*- coding: utf-8 -*-
import os
import pkg_resources

from subprocess import Popen, PIPE, check_output

from ..sysutils import real_path, get_temp_file
from .metric    import Metric

BLEU_SCRIPT = pkg_resources.resource_filename('nmtpy', 'external/multi-bleu.perl')

class BLEUScore(Metric):
    def __init__(self, score=None):
        super(BLEUScore, self).__init__(score)
        self.name = "BLEU"
        if score:
            self.score = float(score.split()[2][:-1])
            self.score_str = score.replace('BLEU = ', '')

"""MultiBleuScorer class."""
class MultiBleuScorer(object):
    def __init__(self, lowercase=False):
        # For multi-bleu.perl we give the reference(s) files as argv,
        # while the candidate translations are read from stdin.
        self.lowercase = lowercase
        self.__cmdline = [BLEU_SCRIPT]
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
