from LibNlp.utils.LibNlpTestCase import LibNlpTestCase
from LibNlp.reader.Reader import Reader
from LibNlp.data.RawDataProcessor import RawDataProcessor
from LibNlp.data.LibDataProcessor import LibDataProcessor
from LibNlp.data.FeatureDict import FeatureDict
from LibNlp.utils.Params import Params


class TestFeatureDict(LibNlpTestCase):
    def setUp(self):
        self.param_path = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/libnlp/libNlp/config/newdefault.json"
        paramController = Params(self.param_path)
        self.args = paramController.args
        self.processor = RawDataProcessor.from_params(self.args.pipeline.data)
        self.processor.load_data(self.args.files.train_file)

        self.featureDict = FeatureDict(
            self.processor.dataset,
            use_qemb=self.args.pipeline.data.params.use_qemb,
            use_in_question=self.args.pipeline.data.params.use_in_question,
            use_pos=self.args.pipeline.data.params.use_pos,
            use_ner=self.args.pipeline.data.params.use_ner,
            use_lemma=self.args.pipeline.data.params.use_lemma,
            use_tf=self.args.pipeline.data.params.use_tf,
        )

    def test_featureDict(self):
        pass
