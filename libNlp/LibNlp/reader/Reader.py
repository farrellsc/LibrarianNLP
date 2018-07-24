from .Model import Model
from utils.TokenDictionary import TokenDictionary
import torch
from torch import optim
import logging


logger = logging.getLogger(__name__)


class Reader:
    """
    The data reader class that handles intializing the underlying model
    architecture, saving, updating examples, and predicting examples.

    """

    def __init__(self, args, word_dict=None, feature_dict=None, state_dict=None):
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
        self.state_dict = state_dict
        self.model = None
        self.optimizer = None
        self.updateCount = 0

    def set_model(self, state_dict=None):
        self.model = Model(self.args.encoding, self.word_dict, self.feature_dict, self.state_dict)

    def init_optimizer(self):
        """
        Initialize an optimizer for the free parameters of the network.

        :return: None
        """
        if self.args.fix_embeddings:
            for p in self.model.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.args.optimizer.type == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.optimizer.learning_rate,
                                       momentum=self.args.optimizer.momentum,
                                       weight_decay=self.args.optimizer.weight_decay)
        elif self.args.optimizer.type == 'adamax':
            self.optimizer = optim.Adamax(parameters, weight_decay=self.args.optimizer.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer.type)

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

    def build_feature_dict(self, examples):
        """Index features (one hot) from fields in examples and options."""

        def _insert(feature):
            if feature not in feature_dict:
                feature_dict[feature] = len(feature_dict)

        feature_dict = {}

        # Exact match features
        if self.args.encoding.params.use_in_question:
            _insert('in_question')
            _insert('in_question_uncased')
            if self.args.encoding.params.use_lemma:
                _insert('in_question_lemma')

        # Part of speech tag features
        if self.args.encoding.params.use_pos:
            for ex in examples:
                for w in ex['pos']:
                    _insert('pos=%s' % w)

        # Named entity tag features
        if self.args.encoding.params.use_ner:
            for ex in examples:
                for w in ex['ner']:
                    _insert('ner=%s' % w)

        # Term frequency feature
        if self.args.encoding.params.use_tf:
            _insert('tf')
        self.feature_dict = feature_dict

    def build_word_dict(self, baseArgs, examples):
        """Return a dictionary from question and document words in
        provided examples.
        """
        word_dict = TokenDictionary()
        for w in self.load_words(baseArgs, examples):
            word_dict.add(w)
        self.word_dict = word_dict

    def load_words(self, baseArgs, examples):
        """Iterate and index all the words in examples (documents + questions)."""

        def _insert(iterable):
            for w in iterable:
                w = TokenDictionary.normalize(w)
                if valid_words and w not in valid_words:
                    continue
                words.add(w)

        if baseArgs.pipeline.dataLoader.restrict_vocab and baseArgs.files.embedding_file:
            logger.info('Restricting to words in %s' % baseArgs.files.embedding_file)
            valid_words = self.index_embedding_words(baseArgs.files.embedding_file)
            logger.info('Num words in set = %d' % len(valid_words))
        else:
            valid_words = None

        words = set()
        for ex in examples:
            _insert(ex['question'])
            _insert(ex['document'])
        return words

    def index_embedding_words(self, embedding_file):
        """Put all the words in embedding_file into a set."""
        words = set()
        with open(embedding_file) as f:
            for line in f:
                w = TokenDictionary.normalize(line.rstrip().split(' ')[0])
                words.add(w)
        return words

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
