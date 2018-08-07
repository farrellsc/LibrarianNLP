from LibNlp.reader.networks.Network import Network
from LibNlp.utils.DotDict import DotDict
from .Model import Model
import torch.nn as nn
from copy import deepcopy


@Model.register("Librarian")
class Librarian(Model):
    """
    The Librarian model is the implementation of DrQA model.
    It is the integration network of 2 RNNs and 2 AttentionNetworks.
    When the model forwards it updates all four networks to generate a prediction.
    """

    def __init__(self, doc_args, question_args, start_aligning_args, end_aligning_args, vocab_size, embedding_dim):
        """
        On initialization 'Model' construct four networks using input args:
            doc_network: StackedBRNN for encoding texts
            question_network: StackedBRNN for encoding question texts
            start_attention: BilinearSeqAttn for Capturing probabilities of token starting / ending positions
            end_attention: BilinearSeqAttn for Capturing probabilities of token starting / ending positions

        :param args: config.pipeline.reader.encoding
        """
        super(Librarian, self).__init__()

        self.doc_args = doc_args
        self.question_args = question_args
        self.start_aligning_args = start_aligning_args
        self.end_aligning_args = end_aligning_args
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # # Projection for attention weighted question
        # if args.use_qemb:
        #     self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)
        #
        # # Input size to RNN: word emb + question emb + manual features
        # doc_input_size = args.embedding_dim + args.num_features
        # if args.use_qemb:
        #     doc_input_size += args.embedding_dim

        self.doc_network = Network.from_params(doc_args)
        self.question_network = Network.from_params(question_args)
        self.start_attention = Network.from_params(start_aligning_args)
        self.end_attention = Network.from_params(end_aligning_args)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

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
    def from_params(cls, model_args: DotDict) -> 'Librarian':
        vocab_size = model_args.pop("vocab_size")
        embedding_dim = model_args.encoding.get("embedding_dim")

        doc_args = deepcopy(model_args.encoding)
        doc_args.pop("question_layers")
        doc_args.num_layers = doc_args.pop('doc_layers')

        question_args = deepcopy(model_args.encoding)
        question_args.pop('doc_layers')
        question_args.num_layers = question_args.pop('question_layers')

        start_aligning_args = model_args.aligning
        end_aligning_args = deepcopy(model_args.aligning)

        return cls(doc_args, question_args, start_aligning_args, end_aligning_args, vocab_size, embedding_dim)
