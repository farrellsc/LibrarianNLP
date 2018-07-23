from .Model import Model
import torch
from torch import optim
import logging


logger = logging.getLogger(__name__)


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
        self.args = args
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.model = Model(word_dict, feature_dict, state_dict)
        raise NotImplementedError

    def init_optimizer(self):
        """
        Initialize an optimizer for the free parameters of the network.

        :return: None
        """
        if self.args.fix_embeddings:
            for p in self.model.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)

    @staticmethod
    def load_checkpoint(filename):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = Reader(args, word_dict, feature_dict, state_dict)
        model.init_optimizer(optimizer)
        return model, epoch

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
