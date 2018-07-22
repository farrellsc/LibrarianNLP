import torch.nn as nn


class Network(nn.Module):
    """
    An abstract ``Network`` class, defines method interfaces for its children classes.
    """

    def __init__(self):
        super(Network, self).__init__()
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError
