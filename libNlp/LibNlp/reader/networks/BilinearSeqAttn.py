import torch.nn as nn
from .Network import Network
from LibNlp.utils.Params import Params


@Network.register("BilinearSeqAttn")
class BilinearSeqAttn(Network):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, doc_hidden_size, question_hidden_size):
        super(BilinearSeqAttn, self).__init__()
        self.linear = nn.Linear(question_hidden_size, doc_hidden_size)
        raise NotImplementedError

    def forward(self, x, y, x_mask):
        """

        :param x: batch * len * hdim1
        :param y: batch * hdim2
        :param x_mask: batch * len (1 for padding, 0 for true)
        :return: alpha = batch * len
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'BilinearSeqAttn':
        doc_hidden_size = params.pop('doc_hidden_size')
        question_hidden_size = params.pop('question_hidden_size')
        params.assert_empty(cls.__name__)
        return cls(doc_hidden_size, question_hidden_size)
