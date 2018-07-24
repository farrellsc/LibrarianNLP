from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from .RawDataProcessor import RawDataProcessor
from LibNlp.utils.Params import Params
from overrides import overrides
from collections import Counter
import json
import torch


@RawDataProcessor.register("LibDataProcessor")
class LibDataProcessor(RawDataProcessor):
    def __init__(self, data_args, batch_size, data_workers):
        self.data_args = data_args
        self.batch_size = batch_size
        self.data_workers = data_workers
        self.model_args = None
        self.word_dict = None
        self.feature_dict = None
        self.dataset = None
        self.loader = None

    def set_loader(self):
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.data_workers,
            collate_fn=self.batchify,
        )

    def load_data(self, filename):
        """Load examples from preprocessed file.
        One example per line, JSON encoded.
        """
        # Load JSON lines
        with open(filename) as f:
            examples = [json.loads(line) for line in f]

        # Make case insensitive?
        if self.args.uncased_question or self.args.uncased_doc:
            for ex in examples:
                if self.args.uncased_question:
                    ex['question'] = [w.lower() for w in ex['question']]
                if self.args.uncased_doc:
                    ex['document'] = [w.lower() for w in ex['document']]
        self.dataset = examples

    @overrides
    def __len__(self):
        return len(self.dataset)

    @overrides
    def __getitem__(self, index):
        return self.vectorize(self.dataset[index])

    def lengths(self):
        return [(len(ex['document']), len(ex['question']))
                for ex in self.dataset]

    def vectorize(self, example):
        """
        Torchify a single example.

        :param example:
        :param model:
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
        if self.model_args.encoding.params.use_in_question:
            q_words_cased = {w for w in example['question']}
            q_words_uncased = {w.lower() for w in example['question']}
            q_lemma = {w for w in example['qlemma']} if self.model_args.encoding.params.use_lemma else None
            for i in range(len(example['document'])):
                if example['document'][i] in q_words_cased:
                    features[i][self.feature_dict['in_question']] = 1.0
                if example['document'][i].lower() in q_words_uncased:
                    features[i][self.feature_dict['in_question_uncased']] = 1.0
                if q_lemma and example['lemma'][i] in q_lemma:
                    features[i][self.feature_dict['in_question_lemma']] = 1.0

        # f_{token} (POS)
        if self.model_args.encoding.params.use_pos:
            for i, w in enumerate(example['pos']):
                f = 'pos=%s' % w
                if f in self.feature_dict:
                    features[i][self.feature_dict[f]] = 1.0

        # f_{token} (NER)
        if self.model_args.encoding.params.use_ner:
            for i, w in enumerate(example['ner']):
                f = 'ner=%s' % w
                if f in self.feature_dict:
                    features[i][self.feature_dict[f]] = 1.0

        # f_{token} (TF)
        if self.model_args.encoding.params.use_tf:
            counter = Counter([w.lower() for w in example['document']])
            l = len(example['document'])
            for i, w in enumerate(example['document']):
                features[i][self.feature_dict['tf']] = counter[w.lower()] * 1.0 / l

        # Maybe return without target
        if 'answers' not in example:
            return document, features, question, example['id']

        # TODO
        # ...or with target(s) (might still be empty if answers is empty)
        if single_answer:
            assert(len(example['answers']) > 0)
            start = torch.LongTensor(1).fill_(example['answers'][0][0])
            end = torch.LongTensor(1).fill_(example['answers'][0][1])
        else:
            start = [a[0] for a in example['answers']]
            end = [a[1] for a in example['answers']]

        return document, features, question, start, end, example['id']

    @staticmethod
    def batchify(self, batch):
        """
        Gather a batch of individual examples into one batch

        :param batch: batch of examples
        :return: x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, ids
        """

        """Gather a batch of individual examples into one batch."""
        NUM_INPUTS = 3
        NUM_TARGETS = 2
        NUM_EXTRA = 1

        ids = [ex[-1] for ex in batch]
        docs = [ex[0] for ex in batch]
        features = [ex[1] for ex in batch]
        questions = [ex[2] for ex in batch]

        # Batch documents and features
        max_length = max([d.size(0) for d in docs])
        x1 = torch.LongTensor(len(docs), max_length).zero_()
        x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
        if features[0] is None:
            x1_f = None
        else:
            x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
        for i, d in enumerate(docs):
            x1[i, :d.size(0)].copy_(d)
            x1_mask[i, :d.size(0)].fill_(0)
            if x1_f is not None:
                x1_f[i, :d.size(0)].copy_(features[i])

        # Batch questions
        max_length = max([q.size(0) for q in questions])
        x2 = torch.LongTensor(len(questions), max_length).zero_()
        x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
        for i, q in enumerate(questions):
            x2[i, :q.size(0)].copy_(q)
            x2_mask[i, :q.size(0)].fill_(0)

        # Maybe return without targets
        if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
            return x1, x1_f, x1_mask, x2, x2_mask, ids

        elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
            # ...Otherwise add targets
            if torch.is_tensor(batch[0][3]):
                y_s = torch.cat([ex[3] for ex in batch])
                y_e = torch.cat([ex[4] for ex in batch])
            else:
                y_s = [ex[3] for ex in batch]
                y_e = [ex[4] for ex in batch]
        else:
            raise RuntimeError('Incorrect number of inputs per example.')

        return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, ids

    @classmethod
    def from_params(cls, params: Params) -> 'LibDataProcessor':
        batch_size = params.pop('batch_size')
        data_workers = params.pop('data_workers')
        data_args = params
        params.assert_empty(cls.__name__)
        return cls(data_args, batch_size, data_workers)