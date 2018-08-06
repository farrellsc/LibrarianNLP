from torch.utils.data import Dataset
from collections import Counter
import torch
import json


class ReaderDataset(Dataset):

    def __init__(
            self,
            uncased_question=False,
            uncased_doc=False,
            use_qemb=False,
            use_in_question=False,
            use_pos=False,
            use_ner=False,
            use_lemma=False,
            use_tf=False
    ):
        self.examples = None
        self.uncased_question = uncased_question
        self.uncased_doc = uncased_doc
        self.use_qemb = use_qemb
        self.use_in_question = use_in_question
        self.use_pos = use_pos
        self.use_ner = use_ner
        self.use_lemma = use_lemma
        self.use_tf = use_tf

        self.word_dict = None
        self.feature_dict = None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.vectorize(self.examples[index])

    def lengths(self):
        return [(len(ex['document']), len(ex['question']))
                for ex in self.examples]

    def set_utils(self, word_dict, feature_dict):
        self.word_dict = word_dict
        self.feature_dict = feature_dict

    def load_data(self, filename):
        """Load examples from preprocessed file.
        One example per line, JSON encoded.
        """
        # Load JSON lines
        with open(filename) as f:
            examples = [json.loads(line) for line in f]

        # Make case insensitive?
        if self.uncased_question or self.uncased_doc:
            for ex in examples:
                if self.uncased_question:
                    ex['question'] = [w.lower() for w in ex['question']]
                if self.uncased_doc:
                    ex['document'] = [w.lower() for w in ex['document']]
        self.examples = examples

    def vectorize(self, example):
        """
        Torchify a single example.

        :param example:
        :return: document, features, question, start, end, ex['id']
        """
        # Index words
        document = torch.LongTensor([self.word_dict[w] for w in example['document']])
        question = torch.LongTensor([self.word_dict[w] for w in example['question']])

        # Create extra features vector
        if len(self.feature_dict) > 0:
            features = torch.zeros(len(example['document']), len(self.feature_dict))
        else:
            features = None

        # f_{exact_match}
        if self.use_in_question:
            q_words_cased = {w for w in example['question']}
            q_words_uncased = {w.lower() for w in example['question']}
            q_lemma = {w for w in example['qlemma']} if self.use_lemma else None
            for i in range(len(example['document'])):
                if example['document'][i] in q_words_cased:
                    features[i][self.feature_dict['in_question']] = 1.0
                if example['document'][i].lower() in q_words_uncased:
                    features[i][self.feature_dict['in_question_uncased']] = 1.0
                if q_lemma and example['lemma'][i] in q_lemma:
                    features[i][self.feature_dict['in_question_lemma']] = 1.0

        # f_{token} (POS)
        if self.use_pos:
            for i, w in enumerate(example['pos']):
                f = 'pos=%s' % w
                if f in self.feature_dict:
                    features[i][self.feature_dict[f]] = 1.0

        # f_{token} (NER)
        if self.use_ner:
            for i, w in enumerate(example['ner']):
                f = 'ner=%s' % w
                if f in self.feature_dict:
                    features[i][self.feature_dict[f]] = 1.0

        # f_{token} (TF)
        if self.use_tf:
            counter = Counter([w.lower() for w in example['document']])
            l = len(example['document'])
            for i, w in enumerate(example['document']):
                features[i][self.feature_dict['tf']] = counter[w.lower()] * 1.0 / l

        # Maybe return without target
        if 'answers' not in example:
            return document, features, question, example['id']

        start = [a[0] for a in example['answers']]
        end = [a[1] for a in example['answers']]

        return document, features, question, start, end, example['id']
