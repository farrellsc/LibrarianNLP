import torch.nn as nn
from .Network import Network
from LibNlp.utils.DotDict import DotDict


@Network.register("BasicLSTM")
class BasicLSTM(Network):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, args: DotDict):
        super(BasicLSTM, self).__init__()
        # TODO
        # construct self.rnns with input params

    def forward(self, data, mask):
        raise NotImplementedError

    @classmethod
    def from_params(cls, args: DotDict) -> 'BasicLSTM':
        return cls(args)
