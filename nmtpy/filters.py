"""Filters out fillers from compound splitted sentences."""
class CompoundFilter(object):
    def __init__(self):
        pass

    def __filter(self, s):
        return s.replace(" @@ ", "").replace(" @@", "").replace(" @", "").replace("@ ", "")

    def __call__(self, inp):
        if isinstance(inp, str):
            return self.__filter(inp)
        else:
            return [self.__filter(e) for e in inp]


__filters = {'compound'     : CompoundFilter()}

def get_filter(name):
    return __filters.get(name, None)
