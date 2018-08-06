from LibNlp.utils.LibNlpTestCase import LibNlpTestCase
from LibNlp.utils.Params import Params


class TestParam(LibNlpTestCase):
    def test_get(self):
        param_path = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/libnlp/libNlp/config/newdefault.json"
        paramController = Params(param_path)
        print(paramController.args)
