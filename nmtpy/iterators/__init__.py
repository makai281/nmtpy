import iter_text, iter_bitext, iter_imgfeats, iter_multi, iter_flickr, iter_wmt16

def get_iterator(name):
    iters = {
                "text"      : iter_text.TextIterator,
                "bitext"    : iter_bitext.BiTextIterator,
                "img_feats" : iter_imgfeats.ImageFeatsIterator,
                "multi"     : iter_multi.MultiIterator,
                "flickr"    : iter_flickr.IterFlickr,
                "wmt16"     : iter_wmt16.WMT16Iterator,
            }
    return iters[name]
