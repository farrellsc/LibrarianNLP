from LibNlp.utils.LibNlpTestCase import LibNlpTestCase
from LibNlp.utils.DotDict import DotDict
from LibNlp.utils.AverageMeter import AverageMeter
from LibNlp.utils.Timer import Timer
from LibNlp.reader.models.Model import Model
from LibNlp.reader.models.Aphorism import Aphorism
from LibNlp.reader.networks.BasicLSTM import BasicLSTM
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import pickle


logger = logging.getLogger(__name__)


class TestAphorism(LibNlpTestCase):
    def setUp(self):
        # --------------------------------------------------------------------------
        # SET args
        args = DotDict({
            "type": "Aphorism"
        })
        self.model = Model.from_params(args)
        self.MODEL_PATH = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/libnlp/libNlp/data/models/Aphorism.pkl"
        assert type(self.model) == Aphorism
        assert type(self.model.main_network) == BasicLSTM

    def test_train_aphorism_pipeline(self):
        """
        In this function you implement Aphorism Model training.

        This is only a suggested formatting from DrQA.script.train.py
        Feel free to change it according to your preference.
        """
        # --------------------------------------------------------------------------
        # SET DATA
        # TODO: Since DataProcessor is still in construction, here please use any method that works
        # TODO:
        data_loader = any_iterable

        # --------------------------------------------------------------------------
        # SET OPTIMIZER
        self.optimizer = optim.SGD(optim_parameters)

        # --------------------------------------------------------------------------
        # Training
        global_timer = Timer()
        for epoch in range(num_epochs):
            train_loss = AverageMeter()
            epoch_time = Timer()
            for idx, example in enumerate(data_loader):
                self.model.train()
                inputs = something_from_example
                target = something_from_example

                score = self.model(*inputs)
                loss = F.nll_loss(score, target)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), grad_clipping)
                self.optimizer.step()

                train_loss.update(loss.data[0], ex[0].size(0))

                if idx % display_iter == 0:
                    logger.info('train: Epoch = %d | iter = %d/%d | ' %
                                (epoch, idx, len(data_loader)) +
                                'loss = %.2f | elapsed time = %.2f (s)' %
                                (train_loss.avg, global_timer.time()))
                    train_loss.reset()

            logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                        (epoch, epoch_time.time()))

        # --------------------------------------------------------------------------
        # Saving
        pickle.dump(self.model, open(self.MODEL_PATH, 'wb'))
