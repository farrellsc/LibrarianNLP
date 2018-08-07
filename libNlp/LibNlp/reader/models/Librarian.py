from LibNlp.reader.networks.Network import Network
from LibNlp.utils.DotDict import DotDict
from .Model import Model
import copy


@Model.register("Librarian")
class Librarian(Model):
    """
    The Librarian model is the implementation of DrQA model.
    It is the integration network of 2 RNNs and 2 AttentionNetworks.
    When the model forwards it updates all four networks to generate a prediction.
    """

    def __init__(self, doc_args, question_args, aligning_args):
        """
        On initialization 'Model' construct four networks using input args:
            doc_network: StackedBRNN for encoding texts
            question_network: StackedBRNN for encoding question texts
            start_attention: BilinearSeqAttn for Capturing probabilities of token starting / ending positions
            end_attention: BilinearSeqAttn for Capturing probabilities of token starting / ending positions

        :param args: config.pipeline.reader.encoding
        """
        self.doc_network = Network.from_params(doc_args)
        self.question_network = Network.from_params(question_args)
        self.start_attention = Network.from_params(aligning_args)
        self.end_attention = Network.from_params(aligning_args)

    def forward(self, doc_word, doc_feature, doc_mask, question_word, question_mask):
        """
        Overwriting 'forward' method in pytorch, used for iterating network. Calls ``forward`` method in all
        sub networks.

        :param doc_word:                    [batch * doc_len]
        :param doc_feature:                 [batch * doc_len * feature_dim]
        :param doc_mask:                    [batch * doc_len]
        :param question_word:               [batch * question_len]
        :param question_mask:               [batch * question_len]
        :return: start_scores, end_scores   []
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, args: DotDict) -> 'Librarian':
        doc_args = copy.deepcopy(args)
        doc_args.encoding.pop("question_layers")
        doc_args.encoding.num_layers = doc_args.encoding.pop('doc_layers')

        question_args = copy.deepcopy(args)
        question_args.encoding.pop('doc_layers')
        question_args.encoding.num_layers = question_args.encoding.pop('question_layers')

        return cls(doc_args, question_args, args.aligning)
