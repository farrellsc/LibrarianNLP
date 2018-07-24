import torch.nn as nn
from LibNlp.utils.Registrable import Registrable
from LibNlp.utils.Params import Params


class Network(nn.Module, Registrable):
    """
    An abstract ``Network`` class, defines method interfaces for its children classes.
    """

    def forward(self, *args):
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'Network':
        # TODO(Mark): The adaptive iterator will need a bit of work here,
        # to retrieve the scaling function etc.

        iterator_type = params.pop_choice("type", cls.list_available())
        return cls.by_name(iterator_type).from_params(params)
