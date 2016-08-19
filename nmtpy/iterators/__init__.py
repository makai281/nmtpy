from wmt            import WMTIterator
from text           import TextIterator
from bitext         import BiTextIterator

def get_iterator(name):
    iters = {
                "wmt"       : WMTIterator,
                "text"      : TextIterator,
                "bitext"    : BiTextIterator,
            }
    return iters[name]
