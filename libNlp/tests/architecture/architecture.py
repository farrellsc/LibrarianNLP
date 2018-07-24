from LibNlp.utils.LibNlpTestCase import LibNlpTestCase
from LibNlp.utils.DotDict import DotDict
from LibNlp.utils.Params import Params
from LibNlp.utils.utils import *
from LibNlp.reader.Reader import Reader


class TestDotDict(LibNlpTestCase):
    def setUp(self):
        configFilePath = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/libnlp/libNlp/config/newdefault.json"
        self.paramController = Params(configFilePath)
        self.args = self.paramController.args
        super(LibNlpTestCase, self).setUp()

    def test_getReader(self):
        reader = init_from_scratch()

