import torch.nn as nn
from LibNlp.utils.Registrable import Registrable
from LibNlp.utils.DotDict import DotDict


class Network(nn.Module, Registrable):
    """
    An abstract ``Network`` class, defines method interfaces for its children classes, including RNNs and
    Attention Mechanisms.
    ``Network`` is a key component to ``Model``.
    """

    def forward(self, *args):
        """
        torch forward method
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, args: DotDict) -> 'Network':
        # to retrieve the scaling function etc.
        iterator_type = args.pop("type")
        return cls.by_name(iterator_type).from_params(args)
