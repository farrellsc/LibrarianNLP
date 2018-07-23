"""
TODO: filter through and restructure these args
"""


import os
import subprocess
import logging
import argparse
from LibNlp import DATA_DIR as LIBNLP_DATA


logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


class Param:
    def __init__(self, parserName):
        self.parser = argparse.ArgumentParser(
            parserName,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.args = None
        self.DATA_DIR = os.path.join(LIBNLP_DATA, 'datasets')
        self.MODEL_DIR = '/tmp/drqa-models/'
        self.EMBED_DIR = os.path.join(LIBNLP_DATA, 'embeddings')

    def set_args(self):
        self.args = self.parser.parse_args()

    def add_pipeline_args(self):
        self.parser.register('type', 'bool', str2bool)

        # Pipeline architecture
        pipeline = self.parser.add_argument_group('DrQA Pipeline Architecture')
        pipeline.add_argument('--reader-model', type=str, default=None,
                            help='Path to trained Document Reader model')
        pipeline.add_argument('--retriever-model', type=str, default=None,
                            help='Path to Document Retriever model (tfidf)')
        pipeline.add_argument('--doc-db', type=str, default=None,
                            help='Path to Document DB')
        pipeline.add_argument('--tokenizer', type=str, default=None,
                            help=("String option specifying tokenizer type to "
                                  "use (e.g. 'corenlp')"))
        pipeline.add_argument('--candidate-file', type=str, default=None,
                            help=("List of candidates to restrict predictions to, "
                                  "one candidate per line"))
        pipeline.add_argument('--no-cuda', action='store_true',
                            help="Use CPU only")
        pipeline.add_argument('--gpu', type=int, default=-1,
                            help="Specify GPU device id to use")

    def add_model_args(self):
        self.parser.register('type', 'bool', str2bool)

        # Model architecture
        model = self.parser.add_argument_group('DrQA Reader Model Architecture')
        model.add_argument('--model-type', type=str, default='rnn',
                           help='Model architecture type')
        model.add_argument('--embedding-dim', type=int, default=300,
                           help='Embedding size if embedding_file is not given')
        model.add_argument('--hidden-size', type=int, default=128,
                           help='Hidden size of RNN units')
        model.add_argument('--doc-layers', type=int, default=3,
                           help='Number of encoding layers for document')
        model.add_argument('--question-layers', type=int, default=3,
                           help='Number of encoding layers for question')
        model.add_argument('--rnn-type', type=str, default='lstm',
                           help='RNN type: LSTM, GRU, or RNN')

        # Model specific details
        detail = self.parser.add_argument_group('DrQA Reader Model Details')
        detail.add_argument('--concat-rnn-layers', type='bool', default=True,
                            help='Combine hidden states from each encoding layer')
        detail.add_argument('--question-merge', type=str, default='self_attn',
                            help='The way of computing the question representation')
        detail.add_argument('--use-qemb', type='bool', default=True,
                            help='Whether to use weighted question embeddings')
        detail.add_argument('--use-in-question', type='bool', default=True,
                            help='Whether to use in_question_* features')
        detail.add_argument('--use-pos', type='bool', default=True,
                            help='Whether to use pos features')
        detail.add_argument('--use-ner', type='bool', default=True,
                            help='Whether to use ner features')
        detail.add_argument('--use-lemma', type='bool', default=True,
                            help='Whether to use lemma features')
        detail.add_argument('--use-tf', type='bool', default=True,
                            help='Whether to use term frequency features')

        # Optimization details
        optim = self.parser.add_argument_group('DrQA Reader Optimization')
        optim.add_argument('--dropout-emb', type=float, default=0.4,
                           help='Dropout rate for word embeddings')
        optim.add_argument('--dropout-rnn', type=float, default=0.4,
                           help='Dropout rate for RNN states')
        optim.add_argument('--dropout-rnn-output', type='bool', default=True,
                           help='Whether to dropout the RNN output')
        optim.add_argument('--optimizer', type=str, default='adamax',
                           help='Optimizer: sgd or adamax')
        optim.add_argument('--learning-rate', type=float, default=0.1,
                           help='Learning rate for SGD only')
        optim.add_argument('--grad-clipping', type=float, default=10,
                           help='Gradient clipping')
        optim.add_argument('--weight-decay', type=float, default=0,
                           help='Weight decay factor')
        optim.add_argument('--momentum', type=float, default=0,
                           help='Momentum factor')
        optim.add_argument('--fix-embeddings', type='bool', default=True,
                           help='Keep word embeddings fixed (use pretrained)')
        optim.add_argument('--tune-partial', type=int, default=0,
                           help='Backprop through only the top N question words')
        optim.add_argument('--rnn-padding', type='bool', default=False,
                           help='Explicitly account for padding in RNN encoding')
        optim.add_argument('--max-len', type=int, default=15,
                           help='The max span allowed during decoding')

    def add_train_args(self):
        """Adds commandline arguments pertaining to training a model. These
        are different from the arguments dictating the model architecture.
        """
        self.parser.register('type', 'bool', str2bool)

        # Runtime environment
        runtime = self.parser.add_argument_group('Environment')
        runtime.add_argument('--no-cuda', type='bool', default=False,
                             help='Train on CPU, even if GPUs are available.')
        runtime.add_argument('--gpu', type=int, default=-1,
                             help='Run on a specific GPU')
        runtime.add_argument('--data-workers', type=int, default=5,
                             help='Number of subprocesses for data loading')
        runtime.add_argument('--parallel', type='bool', default=False,
                             help='Use DataParallel on all available GPUs')
        runtime.add_argument('--random-seed', type=int, default=1013,
                             help=('Random seed for all numpy/torch/cuda '
                                   'operations (for reproducibility)'))
        runtime.add_argument('--num-epochs', type=int, default=40,
                             help='Train data iterations')
        runtime.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for training')
        runtime.add_argument('--test-batch-size', type=int, default=128,
                             help='Batch size during validation/testing')

        # Files
        files = self.parser.add_argument_group('Filesystem')
        files.add_argument('--model-dir', type=str, default=self.MODEL_DIR,
                           help='Directory for saved models/checkpoints/logs')
        files.add_argument('--model-name', type=str, default='',
                           help='Unique model identifier (.mdl, .txt, .checkpoint)')
        files.add_argument('--data-dir', type=str, default=self.DATA_DIR,
                           help='Directory of training/validation data')
        files.add_argument('--train-file', type=str,
                           default='SQuAD-v1.1-train-processed-corenlp.txt',
                           help='Preprocessed train file')
        files.add_argument('--dev-file', type=str,
                           default='SQuAD-v1.1-dev-processed-corenlp.txt',
                           help='Preprocessed dev file')
        files.add_argument('--dev-json', type=str, default='SQuAD-v1.1-dev.json',
                           help=('Unprocessed dev file to run validation '
                                 'while training on'))
        files.add_argument('--embed-dir', type=str, default=self.EMBED_DIR,
                           help='Directory of pre-trained embedding files')
        files.add_argument('--embedding-file', type=str,
                           default='glove.840B.300d.txt',
                           help='Space-separated pretrained embeddings file')

        # Saving + loading
        save_load = self.parser.add_argument_group('Saving/Loading')
        save_load.add_argument('--checkpoint', type='bool', default=False,
                               help='Save model + optimizer state after each epoch')
        save_load.add_argument('--pretrained', type=str, default='',
                               help='Path to a pretrained model to warm-start with')
        save_load.add_argument('--expand-dictionary', type='bool', default=False,
                               help='Expand dictionary of pretrained model to ' +
                                    'include training/dev words of new data')
        # Data preprocessing
        preprocess = self.parser.add_argument_group('Preprocessing')
        preprocess.add_argument('--uncased-question', type='bool', default=False,
                                help='Question words will be lower-cased')
        preprocess.add_argument('--uncased-doc', type='bool', default=False,
                                help='Document words will be lower-cased')
        preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                                help='Only use pre-trained words in embedding_file')

        # General
        general = self.parser.add_argument_group('General')
        general.add_argument('--official-eval', type='bool', default=True,
                             help='Validate with official SQuAD eval')
        general.add_argument('--valid-metric', type=str, default='f1',
                             help='The evaluation metric used for model selection')
        general.add_argument('--display-iter', type=int, default=25,
                             help='Log state after every <display_iter> epochs')
        general.add_argument('--sort-by-len', type='bool', default=True,
                             help='Sort batches by length for speed')

    def add_predict_args(self):
        self.parser.register('type', 'bool', str2bool)

        # prediction args
        parser = argparse.ArgumentParser()
        parser.add_argument('dataset', type=str)
        parser.add_argument('--out-dir', type=str, default='/tmp',
                            help=("Directory to write prediction file to "
                                  "(<dataset>-<model>-pipeline.preds)"))
        parser.add_argument('--reader-model', type=str, default=None,
                            help="Path to trained Document Reader model")
        parser.add_argument('--retriever-model', type=str, default=None,
                            help="Path to Document Retriever model (tfidf)")
        parser.add_argument('--doc-db', type=str, default=None,
                            help='Path to Document DB')
        parser.add_argument('--embedding-file', type=str, default=None,
                            help=("Expand dictionary to use all pretrained "
                                  "embeddings in this file"))
        parser.add_argument('--candidate-file', type=str, default=None,
                            help=("List of candidates to restrict predictions to, "
                                  "one candidate per line"))
        parser.add_argument('--n-docs', type=int, default=5,
                            help="Number of docs to retrieve per query")
        parser.add_argument('--top-n', type=int, default=1,
                            help="Number of predictions to make per query")
        parser.add_argument('--tokenizer', type=str, default=None,
                            help=("String option specifying tokenizer type to use "
                                  "(e.g. 'corenlp')"))
        parser.add_argument('--no-cuda', action='store_true',
                            help="Use CPU only")
        parser.add_argument('--gpu', type=int, default=-1,
                            help="Specify GPU device id to use")
        parser.add_argument('--parallel', action='store_true',
                            help='Use data parallel (split across gpus)')
        parser.add_argument('--num-workers', type=int, default=None,
                            help='Number of CPU processes (for tokenizing, etc)')
        parser.add_argument('--batch-size', type=int, default=128,
                            help='Document paragraph batching size')
        parser.add_argument('--predict-batch-size', type=int, default=1000,
                            help='Question batching size')

    def set_defaults(self):
        """Make sure the commandline arguments are initialized properly."""
        # Check critical files exist
        self.args.dev_json = os.path.join(self.args.data_dir, self.args.dev_json)
        if not os.path.isfile(self.args.dev_json):
            raise IOError('No such file: %s' % self.args.dev_json)
        self.args.train_file = os.path.join(self.args.data_dir, self.args.train_file)
        if not os.path.isfile(self.args.train_file):
            raise IOError('No such file: %s' % self.args.train_file)
        self.args.dev_file = os.path.join(self.args.data_dir, self.args.dev_file)
        if not os.path.isfile(self.args.dev_file):
            raise IOError('No such file: %s' % self.args.dev_file)
        if self.args.embedding_file:
            self.args.embedding_file = os.path.join(self.args.embed_dir, self.args.embedding_file)
            if not os.path.isfile(self.args.embedding_file):
                raise IOError('No such file: %s' % self.args.embedding_file)

        # Set model directory
        subprocess.call(['mkdir', '-p', self.args.model_dir])

        # Set model name
        if not self.args.model_name:
            import uuid
            import time
            self.args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

        # Set log + model file names
        self.args.log_file = os.path.join(self.args.model_dir, self.args.model_name + '.txt')
        self.args.model_file = os.path.join(self.args.model_dir, self.args.model_name + '.mdl')

        # Embeddings options
        if self.args.embedding_file:
            with open(self.args.embedding_file) as f:
                dim = len(f.readline().strip().split(' ')) - 1
            self.args.embedding_dim = dim
        elif not self.args.embedding_dim:
            raise RuntimeError('Either embedding_file or embedding_dim '
                               'needs to be specified.')

        # Make sure tune_partial and fix_embeddings are consistent.
        if self.args.tune_partial > 0 and self.args.fix_embeddings:
            logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
            self.args.fix_embeddings = False

        # Make sure fix_embeddings and embedding_file are consistent
        if self.args.fix_embeddings:
            if not (self.args.embedding_file or self.args.pretrained):
                logger.warning('WARN: fix_embeddings set to False '
                               'as embeddings are random.')
                self.args.fix_embeddings = False
        return self.args
