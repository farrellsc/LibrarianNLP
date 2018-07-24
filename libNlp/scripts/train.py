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
from LibNlp.data.RawDataProcessor import RawDataProcessor
from LibNlp.utils.Params import Params


logger = logging.getLogger()


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx % args.runtime.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.files.checkpoint:
        model.checkpoint(args.files.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


def main(args):
    # --------------------------------------------------------------------------
    # DATA LOADERS
    # Two datasets: train and dev. If we sort by length it's faster.
    # args.pipeline.dataLoader, train_exs, args.pipeline.reader, word_dict, feature_dict, args.pipeline.dataLoader.batch_size, args.runtime.data_workers
    logger.info('-' * 100)
    logger.info('Make data loaders')
    trainProcessor = RawDataProcessor.from_params(args.pipeline.dataLoader)
    devProcessor = RawDataProcessor.from_params(args.pipeline.dataLoader)

    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    trainProcessor.load_data(args.files.train_file)
    logger.info('Num train examples = %d' % len(trainProcessor.dataset))
    devProcessor.load_data(args.files.dev_file)
    logger.info('Num dev examples = %d' % len(devProcessor.dataset))

    # --------------------------------------------------------------------------
    # READER
    # Initialize reader
    logger.info('-' * 100)
    start_epoch = 0
    reader = Reader(args)

    logger.info('-' * 100)
    logger.info('Generate features')
    reader.build_feature_dict(trainProcessor.dataset)
    logger.info('Num features = %d' % len(reader.feature_dict))
    logger.info(reader.feature_dict)

    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build dictionary')
    reader.build_word_dict(args, trainProcessor.dataset + devProcessor.dataset)
    logger.info('Num words = %d' % len(reader.word_dict))

    reader.set_model()

    # Load pretrained embeddings for words in dictionary
    if args.files.embedding_file:
        reader.load_embeddings(reader.word_dict.tokens(), args.files.embedding_file)

    # Set up partial tuning of embeddings
    if args.pipeline.reader.tune_partial > 0:
        logger.info('-' * 100)
        logger.info('Counting %d most frequent question words' % args.pipeline.reader.tune_partial)
        top_words = utils.top_question_words(args, trainProcessor.dataset, reader.word_dict)
        for word in top_words[:5]: logger.info(word)
        logger.info('...')
        for word in top_words[-6:-1]: logger.info(word)
        reader.tune_embeddings([w[0] for w in top_words])

    # Set up optimizer
    reader.init_optimizer()

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
    for epoch in range(start_epoch, args.runtime.num_epochs):
        stats['epoch'] = epoch

        # Train
        train(args, trainProcessor, reader, stats)

        # Validate unofficial (train)
        utils.validate_unofficial(args, trainProcessor, reader, stats, mode='train')

        # Validate unofficial (dev)
        result = utils.validate_unofficial(args, devProcessor, reader, stats, mode='dev')

        # Save best valid
        if result[args.runtime.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.runtime.valid_metric, result[args.runtime.valid_metric],
                         stats['epoch'], reader.updateCount))
            reader.save(args.files.model_file)
            stats['best_valid'] = result[args.runtime.valid_metric]


if __name__ == '__main__':
    # Parse cmdline args and setup environment

    paramController = Params(sys.argv[1])

    # Set random state
    np.random.seed(paramController.args.runtime.random_seed)
    torch.manual_seed(paramController.args.runtime.random_seed)

    # <Comment> Logging
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if paramController.args.files.log_file:
        if paramController.args.files.checkpoint:
            logfile = logging.FileHandler(paramController.args.files.log_file, 'a')
        else:
            logfile = logging.FileHandler(paramController.args.files.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    # </Comment>

    main(paramController.args)
