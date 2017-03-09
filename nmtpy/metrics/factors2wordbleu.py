import os
from subprocess import Popen, PIPE, check_output

from ..sysutils import find_executable, real_path, get_temp_file
from .metric    import Metric
from .bleu   import MultiBleuScorer, BLEUScore

"""Factors2word class."""
class Factors2word(object):
    #def __init__(self, script="factors2word_file.py"):
    def __init__(self, script="factors2word_file_bk.py"):
        self.script = find_executable(script)
        if not self.script:
            raise Exception("factors2word script %s not found." % self.script)


    def compute(self, hyp_file, hyp_mult_file, ref):
        lang = ref.split('.')[-1]
        if lang == 'fr':
            script = "factors2word_file.py"
            self.script = find_executable(script)
        cmdline = ['python', self.script, lang, hyp_file, hyp_mult_file, ref]
        print ('cmdline:', cmdline)

        output = check_output(cmdline)
        out = output.splitlines()
        print (out[0])

        score = out[1].splitlines()
        if len(score) == 0:
            return BLEUScore()
        else:
            return BLEUScore(score[0].rstrip("\n"))

