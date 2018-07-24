from .networks.Network import Network
import copy


class Model:
    """
    The model is the integration network of 2 RNNs and 2 AttentionNetworks.
    When the model forwards it updates all four networks to generate a prediction.
    """

    def __init__(self, args, word_dict, feature_dict, state_dict):
        """

        :param word_dict:
        :param feature_dict:
        :param state_dict:
        """
        self.args = args

        doc_args = copy.deepcopy(self.args)
        doc_args.network.pop("question_layers")
        doc_args.network.num_layers = doc_args.network.pop('doc_layers')
        self.doc_network = Network.from_params(self.args.network)

        question_args = copy.deepcopy(self.args)
        question_args.network.pop('doc_layers')
        question_args.network.num_layers = question_args.network.pop('question_layers')
        self.question_network = Network.from_params(self.args.network)

        self.start_attention = Network.from_params(self.args.aligning)
        self.end_attention = Network.from_params(self.args.aligning)
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
