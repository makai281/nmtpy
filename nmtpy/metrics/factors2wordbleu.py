import os
import subprocess

from ..sysutils import find_executable, real_path, get_temp_file
from .metric    import Metric
from .bleu   import MultiBleuScorer, BLEUScore

"""Factors2word class."""
class Factors2word(object):
    def __init__(self, script="factors2word_file_p3.py"):
        self.script = find_executable(script)
        if not self.script:
            raise Exception("factors2word script %s not found." % self.script)


    def compute(self, hyp_file, hyp_mult_file, ref):
        lang = ref.split('.')[-1]
        cmdline = ['python', self.script, lang, hyp_file, hyp_mult_file, ref]
        print ('cmdline:', cmdline)

        hypstring = None
        with open(hyp_file, "r") as fhyp:
            hypstring = fhyp.read().rstrip()
        
        out = subprocess.run(cmdline, stdout=subprocess.PIPE,
                               input=hypstring, universal_newlines=True).stdout.splitlines()
        print (out[0])

        score = out[1].splitlines()
        if len(score) == 0:
            return BLEUScore()
        else:
            return BLEUScore(score[0].rstrip("\n"))

