from LibNlp.reader.models.Librarian import Model
from LibNlp.utils.TokenDictionary import TokenDictionary
import torch
import copy
from torch import optim
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Reader:
    """
    The data reader class that handles intializing the underlying model
    architecture, saving, updating examples, and predicting examples.

    """

    def __init__(self, word_dict, feature_dict, optimizer_args, model_args, fix_embeddings=False, state_dict=None):
        """
        Build network from word_dict, feature_dict and state_dict

        :param args:
        :param state_dict: neural network states
        """
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.model_args = model_args
        self.model_args.vocab_size = len(word_dict)
        self.model_args.num_features = len(feature_dict)

        self.optimizer_args = optimizer_args
        self.fix_embeddings = fix_embeddings
        self.state_dict = state_dict
        self.model = None
        self.optimizer = None
        self.updateCount = 0

    # --------------------------------------------------------------------------
    # Initiating
    # --------------------------------------------------------------------------

    def set_model(self):
        self.model = Model.from_params(self.model_args)

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.model.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = TokenDictionary.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self, optimizer_state_dict=None):
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

    # --------------------------------------------------------------------------
    # Learning & Predicting
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Loading & Saving
    # --------------------------------------------------------------------------

    @staticmethod
    def load(filename):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        optimizer_args = saved_params['optimizer_args']
        model_args = saved_params['model_args']
        fix_embeddings = saved_params['fix_embeddings']
        epoch = saved_params['epoch']
        optimizer_state_dict = saved_params['optimizer_state_dict']

        reader = Reader(word_dict, feature_dict, optimizer_args, model_args, fix_embeddings, state_dict)
        reader.init_optimizer(optimizer_state_dict)
        return reader, epoch

    def save(self, filename, epoch):
        state_dict = copy.copy(self.model.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'state_dict': self.state_dict,
            'optimizer_args': self.optimizer_args,
            'model_args': self.model_args,
            'fix_embeddings': self.fix_embeddings,
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except Exception:
            logger.warning('WARN: Saving failed... continuing anyway.')