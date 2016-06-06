from iter_wmt import WMTIterator
from iter_text import TextIterator
from iter_bitext import BiTextIterator
from iter_flickr import FlickrIterator
from iter_sqlite import SQLIterator

def get_iterator(name):
    iters = {
                "text"      : TextIterator,
                "bitext"    : BiTextIterator,
                "flickr"    : FlickrIterator,
                "wmt"       : WMTIterator,
                "sqlite"    : SQLIterator,
            }
    return iters[name]
