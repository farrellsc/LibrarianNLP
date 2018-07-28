import torch.nn as nn
from .Network import Network
from LibNlp.utils.DotDict import DotDict


@Network.register("StackedBRNN")
class StackedBRNN(Network):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, embedding_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM):
        """

        :param embedding_size:
        :param hidden_size:
        :param num_layers:
        :param dropout_rate:
        :param dropout_output:
        :param rnn_type:
        """
        super(StackedBRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.dropout_output = dropout_output
        self.rnn_type = rnn_type
        self.rnns = nn.ModuleList()
        # TODO
        # construct self.rnns with input params

    def forward(self, data, mask):
        """
        Encode either padded or non-padded sequences.
        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        :param data:                [batch * len * embedding_dim]
        :param mask:                [batch * len (1 for padding, 0 for true)]
        :return: encoded:           [batch * len * hidden_dim_encoded]
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, args: DotDict) -> 'StackedBRNN':
        rnn_type = args.pop('type')
        embedding_size = args.pop('embedding_dim')
        hidden_size = args.pop('hidden_size')
        num_layers = args.pop('num_layers')
        dropout_rate = args.pop('dropout_rnn', 0)
        dropout_output = args.pop('dropout_rnn_output', False)
        args.assert_empty(cls.__name__)
        return cls(embedding_size, hidden_size, num_layers, dropout_rate=dropout_rate, dropout_output=dropout_output,
                   rnn_type=rnn_type)
