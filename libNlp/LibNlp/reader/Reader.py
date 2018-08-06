from LibNlp.reader.models.Librarian import Model
from LibNlp.utils.TokenDictionary import TokenDictionary
import torch
from torch import optim
import logging


logger = logging.getLogger(__name__)


class Reader:
    """
    The data reader class that handles intializing the underlying model
    architecture, saving, updating examples, and predicting examples.

    """

    def __init__(self, optimizer_args, model_args, fix_embeddings=False, state_dict=None):
        """
        Build network from word_dict, feature_dict and state_dict

        :param args:
        :param state_dict: neural network states
        """
        self.optimizer_args = optimizer_args
        self.model_args = model_args
        self.fix_embeddings = fix_embeddings
        self.state_dict = state_dict
        self.model = None
        self.optimizer = None
        self.updateCount = 0

    def set_model(self, state_dict=None):
        self.model = Model(self.model_args)

    def init_optimizer(self):
        """
        Initialize an optimizer for the free parameters of the network.

        :return: None
        """
        if self.fix_embeddings:
            for p in self.model.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_args.type == 'sgd':
            self.optimizer = optim.SGD(parameters, self.optimizer_args.learning_rate,
                                       momentum=self.optimizer_args.momentum,
                                       weight_decay=self.optimizer_args.weight_decay)
        elif self.optimizer_args.type == 'adamax':
            self.optimizer = optim.Adamax(parameters, weight_decay=self.optimizer_args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.optimizer_args.type)

    @staticmethod
    def load_checkpoint(filename):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        reader = Reader(args, state_dict)
        reader.init_optimizer()
        return reader, epoch

    def update(self, examples):
        """
        Forward a batch of examples; step the optimizer to update weights.

        :param examples: batch of examples
        :return:
        """
        self.updateCount += 1
        raise NotImplementedError

    def predict(self, examples, top_n=1):
        """
        Forward a batch of examples only to get predictions.

        :param examples: batch of examples
        :param top_n: Number of top answers (predictions) to return per batch element.
        :return: prediction
        """
        raise NotImplementedError
