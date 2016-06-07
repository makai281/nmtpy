import os
from subprocess import Popen, PIPE, check_output

from ..sysutils import find_executable, real_path, get_temp_file
from .metric    import Metric

class BLEUScore(Metric):
    def __init__(self, score=None):
        super(BLEUScore, self).__init__(score)
        self.name = "BLEU"
        if score:
            self.score = float(score.split()[2][:-1])
            self.score_str = score.replace('BLEU = ', '')

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
