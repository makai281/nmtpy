#from wmt            import WMTIterator
#from sqlite         import SQLIterator
from text           import TextIterator
#from flickr         import FlickrIterator
from bitext         import BiTextIterator

def get_iterator(name):
    iters = {
                "text"      : TextIterator,
                "bitext"    : BiTextIterator,
            }
    return iters[name]
