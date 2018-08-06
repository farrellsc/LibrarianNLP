from torch.utils.data.dataset import Dataset
from LibNlp.utils.Params import Params
from LibNlp.utils.Registrable import Registrable


class RawDataProcessor(Dataset, Registrable):

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def load_data(self, filename):
        raise NotImplementedError

    def set_utils(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'RawDataProcessor':
        iterator_type = params.pop("type", cls.list_available())
        return cls.by_name(iterator_type).from_params(params)
