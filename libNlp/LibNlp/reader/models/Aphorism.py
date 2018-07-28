from LibNlp.reader.networks.Network import Network
from LibNlp.utils.DotDict import DotDict
from .Model import Model


@Model.register("Aphorism")
class Aphorism(Model):
    """
    The Aphorism model is used for aphorism generation.
    It is implemented by LSTM.
    When the model forwards it updates all four networks to generate a prediction.
    """

    def __init__(self, args: DotDict):
        """
        On initialization 'Model' construct one LSTM.

        :param args: config.pipeline.reader.encoding
        """
        self.args = args

        # TODO: here is just an example
        some_args = {
            "type": "BasicLSTM",
            "hidden_size": 300,
        }
        self.main_network = Network.from_params(DotDict(some_args))

    def forward(self, *inputs):
        """
        Overwriting 'forward' method in pytorch, used for iterating network. Calls ``forward`` method in all
        sub networks.

        :return: score
        """

        # make some preparations
        # lstm_inputs = do_something(inputs)

        # use lstm
        # lstm_outputs = self.main_network(*lstm_inputs)

        # calculate score from lstm_outputs
        # score = do_something(lstm_outputs)

        raise NotImplementedError

    @classmethod
    def from_params(cls, args: DotDict) -> 'Aphorism':
        # raise NotImplementedError
        return cls(args)
