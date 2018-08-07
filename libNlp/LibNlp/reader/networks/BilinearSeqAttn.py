import torch.nn as nn
from .Network import Network
from LibNlp.utils.DotDict import DotDict


@Network.register("BilinearSeqAttn")
class BilinearSeqAttn(Network):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, doc_hidden_size, question_hidden_size):
        """

        :param doc_hidden_size:
        :param question_hidden_size:
        """
        super(BilinearSeqAttn, self).__init__()
        self.linear = nn.Linear(question_hidden_size, doc_hidden_size)

    def forward(self, doc_encoding_result, question_encoding_result, doc_mask):
        """

        :param doc_encoding_result:         [batch * len * doc_hidden_dim]
        :param question_encoding_result:    [batch * question_hidden_dim]
        :param doc_mask:                    [batch * len (1 for padding, 0 for true)]
        :return: alpha:                     [batch * len]
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, args: DotDict) -> 'BilinearSeqAttn':
        doc_hidden_size = args.pop('doc_hidden_size')
        question_hidden_size = args.pop('question_hidden_size')
        return cls(doc_hidden_size, question_hidden_size)
