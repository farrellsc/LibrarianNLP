import torch.nn as nn
from .Network import Network
from LibNlp.utils.Params import Params


@Network.register("StackedBRNN")
class StackedBRNN(Network):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM):
        """

        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param dropout_rate:
        :param dropout_output:
        :param rnn_type:
        """
        super(StackedBRNN, self).__init__()
        self.rnns = nn.ModuleList()
        # construct self.rnns with input params
        raise NotImplementedError

    def forward(self, x, x_mask):
        """
        Encode either padded or non-padded sequences.
        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        :param x: batch * len * hdim
        :param x_mask: batch * len (1 for padding, 0 for true)
        :return: x_encoded: batch * len * hdim_encoded
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'StackedBRNN':
        rnn_type = params.pop('type')
        input_size = params.pop('batch_size')
        hidden_size = params.pop('hidden_size')
        num_layers = params.pop('num_layers')
        dropout_rate = params.pop('dropout_rnn', 0)
        dropout_output = params.pop('dropout_rnn_output', False)
        params.assert_empty(cls.__name__)
        return cls(input_size, hidden_size, num_layers, dropout_rate=dropout_rate, dropout_output=dropout_output,
                   rnn_type=rnn_type)
