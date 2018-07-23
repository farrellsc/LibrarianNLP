"""
TODO:
    validate_official;
    Param section;
    model loading methods: Reader.load_checkpoint, etc.
"""

import logging
import os
import json
import sys
import numpy as np
import torch
from LibNlp.utils import utils
from LibNlp.reader.Reader import Reader
from LibNlp.data.LibDataLoader import LibDataLoader
from LibNlp.utils.Param import Param


logger = logging.getLogger()


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = utils.load_data(args, args.train_file, skip_no_answer=True)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = utils.load_data(args, args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0

    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = Reader.load_checkpoint(checkpoint_file, args)
    elif args.pretrained:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        logger.info('Using pretrained model...')
        model = Reader.load(args.pretrained, args)
        if args.expand_dictionary:
            logger.info('Expanding dictionary for new data...')
            # Add words in training + dev examples
            words = utils.load_words(args, train_exs + dev_exs)
            added = model.expand_dictionary(words)
            # Load pretrained embeddings for added words
            if args.embedding_file:
                model.load_embeddings(added, args.embedding_file)

        # Set up partial tuning of embeddings
        if args.tune_partial > 0:
            logger.info('-' * 100)
            logger.info('Counting %d most frequent question words' % args.tune_partial)

            top_words = utils.top_question_words(args, train_exs, model.word_dict)

            for word in top_words[:5]: logger.info(word)
            logger.info('...')
            for word in top_words[-6:-1]: logger.info(word)

            model.tune_embeddings([w[0] for w in top_words])

        # Set up optimizer
        model.init_optimizer()
    else:
        # a brand new reader
        logger.info('Training model from scratch...')
        model = utils.init_from_scratch(args, train_exs, dev_exs)

        # Set up partial tuning of embeddings
        if args.tune_partial > 0:
            logger.info('-' * 100)
            logger.info('Counting %d most frequent question words' % args.tune_partial)

            top_words = utils.top_question_words(args, train_exs, model.word_dict)

            for word in top_words[:5]: logger.info(word)
            logger.info('...')
            for word in top_words[-6:-1]: logger.info(word)

            model.tune_embeddings([w[0] for w in top_words])

        # Set up optimizer
        model.init_optimizer()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')

    train_loader = LibDataLoader(train_exs, model, args.batch_size, args.data_workers)
    dev_loader = LibDataLoader(dev_exs, model, args.test_batch_size, args.data_workers)

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        train(args, train_loader, model, stats)

        # Validate unofficial (train)
        utils.validate_unofficial(args, train_loader, model, stats, mode='train')

        # Validate unofficial (dev)
        result = utils.validate_unofficial(args, dev_loader, model, stats, mode='dev')

        # Save best valid
        if result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]


if __name__ == '__main__':
    # Parse cmdline args and setup environment

    paramController = Param('LibNlp Document Reader')
    paramController.add_train_args()
    paramController.add_model_args()
    paramController.set_args()
    paramController.set_defaults()

    # Set random state
    np.random.seed(paramController.args.random_seed)
    torch.manual_seed(paramController.args.random_seed)

    # <Comment> Non-Essential
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if paramController.args.log_file:
        if paramController.args.checkpoint:
            logfile = logging.FileHandler(paramController.args.log_file, 'a')
        else:
            logfile = logging.FileHandler(paramController.args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    # </Comment>

    main(paramController.args)
