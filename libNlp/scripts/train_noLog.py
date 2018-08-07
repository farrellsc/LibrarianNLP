import sys
import numpy as np
import torch
from LibNlp.utils import utils
from LibNlp.reader.Reader import Reader
from LibNlp.data.RawDataProcessor import RawDataProcessor
from LibNlp.utils.Params import Params
from LibNlp.data.FeatureDict import FeatureDict
from LibNlp.data.WordDict import WordDict


def train(args, dataProcessor, model):
    train_loss = utils.AverageMeter()
    for idx, ex in enumerate(dataProcessor):
        train_loss.update(*model.update(ex))
        if idx % args.runtime.display_iter == 0:
            train_loss.reset()


def main(args):
    trainProcessor = RawDataProcessor.from_params(args.pipeline.data)
    devProcessor = RawDataProcessor.from_params(args.pipeline.data)
    trainProcessor.load_data(args.files.train_file)
    devProcessor.load_data(args.files.dev_file)

    featureDict = FeatureDict(
        trainProcessor.dataset,
        use_qemb=args.pipeline.data.params.use_qemb,
        use_in_question=args.pipeline.data.params.use_in_question,
        use_pos=args.pipeline.data.params.use_pos,
        use_ner=args.pipeline.data.params.use_ner,
        use_lemma=args.pipeline.data.params.use_lemma,
        use_tf=args.pipeline.data.params.use_tf,
    )
    wordDict = WordDict(
        trainProcessor.dataset + devProcessor.dataset,
        embedding_file=args.files.embedding_file,
        restrict_vocab=args.pipeline.data.dataProcessor.restrict_vocab
    )
    trainProcessor.set_utils(word_dict=wordDict, feature_dict=featureDict)
    devProcessor.set_utils(word_dict=wordDict, feature_dict=featureDict)

    # --------------------------------------------------------------------------------------------
    # Data section Above finished & tested
    # --------------------------------------------------------------------------------------------

    reader = Reader(
        args.pipeline.reader.optimizer,
        args.pipeline.reader.model,
        fix_embeddings=args.pipeline.fix_embeddings
    )
    reader.set_model()

    if args.files.embedding_file:
        reader.load_embeddings(reader.word_dict.tokens(), args.files.embedding_file)
    if args.pipeline.reader.tune_partial > 0:
        top_words = utils.top_question_words(args, trainProcessor.dataset, wordDict)
        reader.tune_embeddings([w[0] for w in top_words])
    reader.init_optimizer()

    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    start_epoch = 0
    for epoch in range(start_epoch, args.runtime.num_epochs):
        stats['epoch'] = epoch
        train(args, trainProcessor, reader, stats)
        utils.validate_unofficial(args, trainProcessor, reader, stats, mode='train')
        result = utils.validate_unofficial(args, devProcessor, reader, stats, mode='dev')
        if result[args.runtime.valid_metric] > stats['best_valid']:
            reader.save(args.files.model_file)
            stats['best_valid'] = result[args.runtime.valid_metric]


if __name__ == '__main__':
    paramController = Params(sys.argv[1])
    np.random.seed(paramController.args.runtime.random_seed)
    torch.manual_seed(paramController.args.runtime.random_seed)
    main(paramController.args)
