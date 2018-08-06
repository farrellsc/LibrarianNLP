from LibNlp.utils.LibNlpTestCase import LibNlpTestCase
from LibNlp.reader.Reader import Reader
from LibNlp.data.RawDataProcessor import RawDataProcessor
from LibNlp.data.LibDataProcessor import LibDataProcessor
from LibNlp.data.WordDict import WordDict
from LibNlp.utils.Params import Params


class TestWordDict(LibNlpTestCase):
    def setUp(self):
        self.param_path = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/libnlp/libNlp/config/newdefault.json"
        paramController = Params(self.param_path)
        self.args = paramController.args
        self.processor = RawDataProcessor.from_params(self.args.pipeline.data)
        self.processor.load_data(self.args.files.train_file)

        self.wordDict = WordDict(
            self.processor.dataset,
            embedding_file=self.args.files.embedding_file,
            restrict_vocab=self.args.pipeline.data.dataProcessor.restrict_vocab
        )

    def test_wordDict(self):
        pass
