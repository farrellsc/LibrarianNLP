from LibNlp.utils.DotDict import DotDict
from LibNlp.utils.Registrable import Registrable
import torch.nn as nn


class Model(nn.Module, Registrable):
    """
    An abstract ``Model`` class, defines the compositions for neural networks.
    """

    def __init__(self, model_args):
        self.model_args = model_args

    def forward(self, *inputs, **kwargs):
        """
        Overwriting 'forward' method in pytorch, used for iterating network. Calls ``forward`` method in all
        sub networks.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, args: DotDict) -> 'Model':
        # to retrieve the scaling function etc.
        iterator_type = args.get("type", cls.list_available())
        Registrable.list_available()
        return cls.by_name(iterator_type).from_params(args)
