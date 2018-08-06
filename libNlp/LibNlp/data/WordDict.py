from LibNlp.utils.TokenDictionary import TokenDictionary
import logging

logger = logging.getLogger(__name__)


class WordDict(dict):

    def __init__(self, examples, embedding_file=False, restrict_vocab=False):
        self.embedding_file = embedding_file
        self.restrict_vocab = restrict_vocab
        self.word_dict = self.build_word_dict(examples)

    def __getitem__(self, index):
        return self.word_dict[index]

    def __len__(self):
        return len(self.word_dict)

    def build_word_dict(self, examples):
        """Return a dictionary from question and document words in
        provided examples.
        """
        word_dict = TokenDictionary()
        for w in self.load_words(examples):
            word_dict.add(w)
        return word_dict

    def load_words(self, examples):
        """Iterate and index all the words in examples."""

        def _insert(iterable):
            for w in iterable:
                w = TokenDictionary.normalize(w)
                if valid_words and w not in valid_words:
                    continue
                words.add(w)

        if self.restrict_vocab and self.embedding_file:
            logger.info('Restricting to words in %s' % self.embedding_file)
            valid_words = self.index_embedding_words(self.embedding_file)
            logger.info('Num words in set = %d' % len(valid_words))
        else:
            valid_words = None

        words = set()
        for ex in examples.examples:
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
