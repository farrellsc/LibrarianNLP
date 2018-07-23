from .networks.StackedBRNN import StackedBRNN
from .networks.BilinearSeqAttn import BilinearSeqAttn


class Model:
    """
    The model is the integration network of 2 RNNs and 2 AttentionNetworks.
    When the model forwards it updates all four networks to generate a prediction.
    """

    def __init__(self, word_dict, feature_dict, state_dict):
        """

        :param word_dict:
        :param feature_dict:
        :param state_dict:
        """
        self.doc_network = StackedBRNN(*doc_network_args)
        self.question_network = StackedBRNN(*question_network_args)
        self.start_attention = BilinearSeqAttn(*start_attention_args)
        self.end_attention = BilinearSeqAttn(*end_attention_args)
        raise NotImplementedError

    def forward(self, doc_word, doc_feature, doc_mask, question_word, question_mask):
        """
        overwriting 'forward' method in pytorch, used for iterating network

        :param doc_word:
        :param doc_feature:
        :param doc_mask:
        :param question_word:
        :param question_mask:
        :return: start_scores, end_scores
        """
        raise NotImplementedError
