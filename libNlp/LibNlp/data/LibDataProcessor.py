from torch.utils.data.dataloader import DataLoader
from .RawDataProcessor import RawDataProcessor
from .ReaderDataset import ReaderDataset
from LibNlp.utils.Params import Params
from overrides import overrides
import torch


@RawDataProcessor.register("LibDataProcessor")
class LibDataProcessor(RawDataProcessor):
    def __init__(self, batch_size, data_workers, dataset_args):
        self.batch_size = batch_size
        self.data_workers = data_workers
        self.dataset = ReaderDataset(
            uncased_question=dataset_args.dataProcessor.uncased_question,
            uncased_doc=dataset_args.dataProcessor.uncased_doc,
            use_qemb=dataset_args.params.use_qemb,
            use_in_question=dataset_args.params.use_in_question,
            use_pos=dataset_args.params.use_pos,
            use_ner=dataset_args.params.use_ner,
            use_lemma=dataset_args.params.use_lemma,
            use_tf=dataset_args.params.use_tf
        )
        self.loader = None
        self.word_dict = None
        self.feature_dict = None

    def set_loader(self):
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.args.dataProcessor.data_workers,
            collate_fn=self.batchify,
        )

    def load_data(self, filename):
        self.dataset.load_data(filename)

    @overrides
    def __len__(self):
        return len(self.dataset)

    @overrides
    def __getitem__(self, index):
        return self.vectorize(self.dataset[index])

    def lengths(self):
        return [(len(ex['document']), len(ex['question']))
                for ex in self.dataset]

    def set_utils(self, word_dict, feature_dict):
        self.dataset.set_utils(word_dict, feature_dict)

    @staticmethod
    def batchify(batch):
        """
        Gather a batch of individual examples into one batch

        :param batch: batch of examples
        :return: docs_data, x1_f, docs_mask, questions_data, questions_mask, start, end, ids
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
        docs_indices = torch.LongTensor(len(docs), max_length).zero_()
        docs_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
        if features[0] is None:
            docs_feature = None
        else:
            docs_feature = torch.zeros(len(docs), max_length, features[0].size(1))
        for i, d in enumerate(docs):
            docs_indices[i, :d.size(0)].copy_(d)
            docs_mask[i, :d.size(0)].fill_(0)
            if docs_feature is not None:
                docs_feature[i, :d.size(0)].copy_(features[i])

        # Batch questions
        max_length = max([q.size(0) for q in questions])
        questions_indices = torch.LongTensor(len(questions), max_length).zero_()
        questions_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
        for i, q in enumerate(questions):
            questions_indices[i, :q.size(0)].copy_(q)
            questions_mask[i, :q.size(0)].fill_(0)

        # Maybe return without targets
        if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
            return docs_indices, docs_feature, docs_mask, questions_indices, questions_mask, ids

        elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
            # ...Otherwise add targets
            if torch.is_tensor(batch[0][3]):
                start = torch.cat([ex[3] for ex in batch])
                end = torch.cat([ex[4] for ex in batch])
            else:
                start = [ex[3] for ex in batch]
                end = [ex[4] for ex in batch]
        else:
            raise RuntimeError('Incorrect number of inputs per example.')

        return docs_indices, docs_feature, docs_mask, questions_indices, questions_mask, start, end, ids

    @classmethod
    def from_params(cls, params: Params) -> 'LibDataProcessor':
        return cls(
            params.dataProcessor.batch_size,
            params.dataProcessor.data_workers,
            params
        )
