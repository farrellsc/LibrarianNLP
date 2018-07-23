"""
TODO:
    validate_official;
    config section;
    model loading methods: Reader.load_checkpoint
"""

import os
import numpy as np
import torch
from LibNlp.utils import utils
from LibNlp.reader.Reader import Reader
from LibNlp.data.LibDataLoader import LibDataLoader
from LibNlp.utils.Param import Param


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:
            train_loss.reset()

    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    train_exs = utils.load_data(args, args.train_file, skip_no_answer=True)
    dev_exs = utils.load_data(args, args.dev_file)

    # --------------------------------------------------------------------------
    # MODEL
    start_epoch = 0

    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = Reader.load_checkpoint(checkpoint_file, args)
    elif args.pretrained:
        model = Reader.load(args.pretrained, args)
        if args.expand_dictionary:
            words = utils.load_words(args, train_exs + dev_exs)
            added = model.expand_dictionary(words)
            if args.embedding_file:
                model.load_embeddings(added, args.embedding_file)
        if args.tune_partial > 0:
            top_words = utils.top_question_words(args, train_exs, model.word_dict)
            model.tune_embeddings([w[0] for w in top_words])
        model.init_optimizer()
    else:
        model = utils.init_from_scratch(args, train_exs, dev_exs)
        if args.tune_partial > 0:
            top_words = utils.top_question_words(args, train_exs, model.word_dict)
            model.tune_embeddings([w[0] for w in top_words])
        model.init_optimizer()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    train_loader = LibDataLoader(train_exs, model, args.batch_size, args.data_workers)
    dev_loader = LibDataLoader(dev_exs, model, args.test_batch_size, args.data_workers)

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch
        train(args, train_loader, model, stats)
        utils.validate_unofficial(args, train_loader, model, stats, mode='train')
        result = utils.validate_unofficial(args, dev_loader, model, stats, mode='dev')
        if result[args.valid_metric] > stats['best_valid']:
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]


if __name__ == '__main__':

    paramController = Param('LibNlp Document Reader')
    paramController.add_train_args()
    paramController.add_model_args()
    paramController.set_args()
    paramController.set_defaults()

    np.random.seed(paramController.args.random_seed)
    torch.manual_seed(paramController.args.random_seed)

    main(paramController.args)
