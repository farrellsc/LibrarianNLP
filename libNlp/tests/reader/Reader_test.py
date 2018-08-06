from LibNlp.utils.LibNlpTestCase import LibNlpTestCase
from LibNlp.reader.Reader import Reader
from LibNlp.data.RawDataProcessor import RawDataProcessor
from LibNlp.data.LibDataProcessor import LibDataProcessor
from LibNlp.utils.Params import Params


class TestReader(LibNlpTestCase):
    def setUp(self):
        self.param_path = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/libnlp/libNlp/config/newdefault.json"
        paramController = Params(self.param_path)
        self.args = paramController.args
        self.processor = RawDataProcessor.from_params(self.args.pipeline.data)
        self.reader = Reader(self.args)
