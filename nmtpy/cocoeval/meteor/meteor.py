#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import sys
import subprocess
import threading
import pkg_resources

METEOR_JAR = pkg_resources.resource_filename('nmtpy', 'meteor_data/meteor-1.5.jar')

class Meteor:
    def __init__(self, language, norm=False):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, '-', '-', '-stdio', '-l', language]
        self.env = os.environ
        self.env['LC_ALL'] = 'en_US.UTF_8'

        if norm:
            self.meteor_cmd.append('-norm')

        self.meteor_p = subprocess.Popen(self.meteor_cmd, stdin=subprocess.PIPE, \
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        env=self.env)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def method(self):
        return "METEOR"

    def compute_score(self, gts, res):
        imgIds = sorted(gts.keys())
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        # Send to METEOR
        self.meteor_p.stdin.write('{}\n'.format(eval_line))

        # Collect segment scores
        for i in range(len(imgIds)):
            score = float(self.meteor_p.stdout.readline().strip())
            scores.append(score)

        # Final score
        final_score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return final_score, scores

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        # We obtained --> SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        return self.meteor_p.stdout.readline().strip()

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.wait()
        self.lock.release()
