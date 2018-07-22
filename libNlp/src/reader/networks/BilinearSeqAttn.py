import torch.nn as nn
from .Network import Network


class BilinearSeqAttn(Network):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size):
        super(BilinearSeqAttn, self).__init__()
        self.linear = nn.Linear(y_size, x_size)
        raise NotImplementedError

    def forward(self, x, y, x_mask):
        """

        :param x: batch * len * hdim1
        :param y: batch * hdim2
        :param x_mask: batch * len (1 for padding, 0 for true)
        :return: alpha = batch * len
        """
        raise NotImplementedError

