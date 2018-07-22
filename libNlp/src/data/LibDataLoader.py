from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class LibDataLoader(Dataset):
    def __init__(self, examples, model, batch_size, data_workers):
        self.model = model
        self.dataset = examples

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=data_workers,
            collate_fn=self.batchify,
        )
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.vectorize(self.dataset[index], self.model)

    def lengths(self):
        return [(len(ex['document']), len(ex['question']))
                for ex in self.dataset]

    def vectorize(self, example, model):
        """
        Torchify a single example.

        :param example:
        :param model:
        :return: document, features, question, start, end, ex['id']
        """
        raise NotImplementedError

    def batchify(self, batch):
        """
        Gather a batch of individual examples into one batch

        :param batch: batch of examples
        :return: x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, ids
        """
        raise NotImplementedError