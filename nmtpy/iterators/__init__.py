from iter_wmt import WMTIterator
from iter_text import TextIterator
from iter_bitext import BiTextIterator
from iter_flickr import IterFlickr
from iter_imgfeats import ImageFeatsIterator

def get_iterator(name):
    iters = {
                "text"      : TextIterator,
                "bitext"    : BiTextIterator,
                "img_feats" : ImageFeatsIterator,
                "flickr"    : IterFlickr,
                "wmt"       : WMTIterator,
            }
    return iters[name]
