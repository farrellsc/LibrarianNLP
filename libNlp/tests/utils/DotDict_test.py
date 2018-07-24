from LibNlp.utils.LibNlpTestCase import LibNlpTestCase
from LibNlp.utils.DotDict import DotDict


class TestDotDict(LibNlpTestCase):
    def test_get(self):
        d = {'a': 1, 'b': {'c': 2, 'd': 3}}
        dd = DotDict(d)
        assert dd.a == 1
        assert dd.b.c == 2
        assert dd.b.d == 3

    def test_add(self):
        d = {'a': 1, 'b': {'c': 2, 'd': 3}}
        dd = DotDict(d)
        dd.b.e = 4
        assert dd.b.e == 4

