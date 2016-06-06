from functools import total_ordering

from bleu   import MultiBleuScorer
from meteor import METEORScorer

@total_ordering
class Metric(object):
    def __init__(self, score=None):
        pass

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return "%s = %s" % (self.name, self.__score_str)

def get_scorer(scorer):
    scorers = {
                'meteor': METEORScorer,
                'bleu'  : MultiBleuScorer,
              }

    if scorer == 'all':
        return scorers
    else:
        return scorers[scorer]
