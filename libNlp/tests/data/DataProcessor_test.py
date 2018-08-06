from LibNlp.utils.LibNlpTestCase import LibNlpTestCase
from LibNlp.data.RawDataProcessor import RawDataProcessor
from LibNlp.data.LibDataProcessor import LibDataProcessor
from LibNlp.utils.Params import Params
from LibNlp.data.FeatureDict import FeatureDict
from LibNlp.data.WordDict import WordDict


class TestDataProcessor(LibNlpTestCase):
    def setUp(self):
        self.param_path = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/libnlp/libNlp/config/newdefault.json"
        paramController = Params(self.param_path)
        self.args = paramController.args
        self.processor = RawDataProcessor.from_params(self.args.pipeline.data)

    def test_from_params(self):
        assert type(self.processor) == LibDataProcessor

    def test_load_data(self):
        self.processor.load_data(self.args.files.train_file)
        featureDict = FeatureDict(
            self.processor.dataset,
            use_qemb=self.args.pipeline.data.params.use_qemb,
            use_in_question=self.args.pipeline.data.params.use_in_question,
            use_pos=self.args.pipeline.data.params.use_pos,
            use_ner=self.args.pipeline.data.params.use_ner,
            use_lemma=self.args.pipeline.data.params.use_lemma,
            use_tf=self.args.pipeline.data.params.use_tf,
        )
        wordDict = WordDict(
            self.processor.dataset,
            embedding_file=self.args.files.embedding_file,
            restrict_vocab=self.args.pipeline.data.dataProcessor.restrict_vocab
        )
        self.processor.set_utils(feature_dict=featureDict, word_dict=wordDict)
        assert type(self.processor.dataset.feature_dict) == FeatureDict
        assert type(self.processor.dataset.word_dict) == WordDict
        assert len(self.processor.dataset[0]) == 6
