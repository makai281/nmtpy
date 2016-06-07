from bleu   import MultiBleuScorer
from meteor import METEORScorer

def get_scorer(scorer):
    scorers = {
                'meteor': METEORScorer,
                'bleu'  : MultiBleuScorer,
              }

    if scorer == 'all':
        return scorers
    else:
        return scorers[scorer]
