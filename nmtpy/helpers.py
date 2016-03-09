class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == '__main__':
    d = DotDict()

    d['a'] = 3
    assert d.a == 3
