from LibNlp.utils.LibNlpTestCase import LibNlpTestCase
from LibNlp.utils.DotDict import DotDict
from copy import deepcopy


class TestDotDict(LibNlpTestCase):
    def setUp(self):
        self.d = {'a': 1, 'b': {'c': 2, 'd': 3}}
        self.dd = DotDict(self.d)

    def test_get(self):
        assert self.dd.a == 1
        assert self.dd.b.c == 2
        assert self.dd.b.d == 3

    def test_add(self):
        self.dd.b.e = 4
        assert self.dd.b.e == 4

    def test_deepcopy(self):
        ddd = deepcopy(self.dd)
        print(ddd)
        assert id(self.dd) != id(ddd)
