from .Model import Model


class Reader:
    """
    The data reader class that handles intializing the underlying model
    architecture, saving, updating examples, and predicting examples.

    """

    def __init__(self, args, word_dict, feature_dict, state_dict=None):
        """
        Build network from word_dict, feature_dict and state_dict

        :param args:
        :param word_dict: Dictionary of words
        :param feature_dict: Dictionary of features
        :param state_dict: neural network states
        """
        self.model = Model(word_dict, feature_dict, state_dict)
        raise NotImplementedError

    def update(self, exmaples):
        """
        Forward a batch of examples; step the optimizer to update weights.

        :param exmaples: batch of examples
        :return:
        """
        raise NotImplementedError

    def predict(self, examples, top_n=1):
        """
        Forward a batch of examples only to get predictions.

        :param examples: batch of examples
        :param top_n: Number of top answers (predictions) to return per batch element.
        :return: prediction
        """
        raise NotImplementedError
