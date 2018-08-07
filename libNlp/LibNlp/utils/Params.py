"""
TODO: filter through and restructure these args
"""


import os
import subprocess
import logging
import argparse
import copy
import json
from urllib.parse import urlparse
from typing import Any, Dict, List
from collections import MutableMapping
from overrides import overrides
from .ConfigurationError import ConfigurationError
from .DotDict import DotDict
from LibNlp import DATA_DIR as LIBNLP_DATA


logger = logging.getLogger()


class Params(MutableMapping):
    """
    Represents a parameter dictionary with a history, and contains other functionality around
    parameter passing and validation for AllenNLP.

    There are currently two benefits of a ``Params`` object over a plain dictionary for parameter
    passing:

    #. We handle a few kinds of parameter validation, including making sure that parameters
       representing discrete choices actually have acceptable values, and making sure no extra
       parameters are passed.
    #. We log all parameter reads, including default values.  This gives a more complete
       specification of the actual parameters used than is given in a JSON / HOCON file, because
       those may not specify what default values were used, whereas this will log them.

    The convention for using a ``Params`` object in AllenNLP is that you will consume the parameters
    as you read them, so that there are none left when you've read everything you expect.  This
    lets us easily validate that you didn't pass in any `extra` parameters, just by making sure
    that the parameter dictionary is empty.  You should do this when you're done handling
    parameters, by calling :func:`Params.assert_empty`.
    """
    DEFAULT = object()

    def __init__(
            self,
            params_file: str,
            # history: str = "",
            # loading_from_archive: bool = False,
            # files_to_archive: Dict[str, str] = None
    ) -> None:
        self.args = DotDict(_replace_none(json.load(open(params_file))))
        # self.history = history
        # self.loading_from_archive = loading_from_archive
        # self.files_to_archive = {} if files_to_archive is None else files_to_archive
        self.set_defaults()

    def add_file_to_archive(self, name: str) -> None:
        """
        Any class in its ``from_params`` method can request that some of its
        input files be added to the archive by calling this method.

        For example, if some class ``A`` had an ``input_file`` parameter, it could call

        ```
        params.add_file_to_archive("input_file")
        ```

        which would store the supplied value for ``input_file`` at the key
        ``previous.history.and.then.input_file``. The ``files_to_archive`` dict
        is shared with child instances via the ``_check_is_dict`` method, so that
        the final mapping can be retrieved from the top-level ``Params`` object.

        NOTE: You must call ``add_file_to_archive`` before you ``pop()``
        the parameter, because the ``Params`` instance looks up the value
        of the filename inside itself.

        If the ``loading_from_archive`` flag is True, this will be a no-op.
        """
        if not self.loading_from_archive:
            self.files_to_archive[f"{self.history}{name}"] = self.get(name)

    @overrides
    def pop(self, key: str, default: Any = DEFAULT):
        """
        Performs the functionality associated with dict.pop(key), along with checking for
        returned dictionaries, replacing them with Param objects with an updated history.

        If ``key`` is not present in the dictionary, and no default was specified, we raise a
        ``ConfigurationError``, instead of the typical ``KeyError``.
        """
        if default is self.DEFAULT:
            try:
                value = self.args.pop(key)
            except KeyError:
                raise ConfigurationError("key \"{}\" is required at location \"{}\"".format(key, self.history))
        else:
            value = self.args.pop(key, default)
        if not isinstance(value, dict):
            logger.info(self.history + key + " = " + str(value))  # type: ignore
        return self._check_is_dict(key, value)

    @overrides
    def get(self, key: str, default: Any = DEFAULT):
        """
        Performs the functionality associated with dict.get(key) but also checks for returned
        dicts and returns a Params object in their place with an updated history.
        """
        if default is self.DEFAULT:
            try:
                value = self.args.get(key)
            except KeyError:
                raise ConfigurationError("key \"{}\" is required at location \"{}\"".format(key, self.history))
        else:
            value = self.args.get(key, default)
        return self._check_is_dict(key, value)

    def pop_choice(self, key: str, choices: List[Any], default_to_first_choice: bool = False):
        """
        Gets the value of ``key`` in the ``params`` dictionary, ensuring that the value is one of
        the given choices. Note that this `pops` the key from params, modifying the dictionary,
        consistent with how parameters are processed in this codebase.

        Parameters
        ----------
        key: str
            Key to get the value from in the param dictionary
        choices: List[Any]
            A list of valid options for values corresponding to ``key``.  For example, if you're
            specifying the type of encoder to use for some part of your model, the choices might be
            the list of encoder classes we know about and can instantiate.  If the value we find in
            the param dictionary is not in ``choices``, we raise a ``ConfigurationError``, because
            the user specified an invalid value in their parameter file.
        default_to_first_choice: bool, optional (default=False)
            If this is ``True``, we allow the ``key`` to not be present in the parameter
            dictionary.  If the key is not present, we will use the return as the value the first
            choice in the ``choices`` list.  If this is ``False``, we raise a
            ``ConfigurationError``, because specifying the ``key`` is required (e.g., you `have` to
            specify your model class when running an experiment, but you can feel free to use
            default settings for encoders if you want).
        """
        default = choices[0] if default_to_first_choice else self.DEFAULT
        value = self.pop(key, default)
        if value not in choices:
            key_str = self.history + key
            message = '%s not in acceptable choices for %s: %s' % (value, key_str, str(choices))
            raise ConfigurationError(message)
        return value

    def as_dict(self, quiet=False):
        """
        Sometimes we need to just represent the parameters as a dict, for instance when we pass
        them to a Keras layer(so that they can be serialised).

        Parameters
        ----------
        quiet: bool, optional (default = False)
            Whether to log the parameters before returning them as a dict.
        """
        if quiet:
            return self.args

        def log_recursively(parameters, history):
            for key, value in parameters.items():
                if isinstance(value, dict):
                    new_local_history = history + key  + "."
                    log_recursively(value, new_local_history)
                else:
                    logger.info(history + key + " = " + str(value))

        logger.info("Converting Params object to dict; logging of default "
                    "values will not occur when dictionary parameters are "
                    "used subsequently.")
        logger.info("CURRENTLY DEFINED PARAMETERS: ")
        log_recursively(self.args, self.history)
        return self.args

    def duplicate(self) -> 'Params':
        """
        Uses ``copy.deepcopy()`` to create a duplicate (but fully distinct)
        copy of these Params.
        """
        return Params(copy.deepcopy(self.args))

    def assert_empty(self, class_name: str):
        """
        Raises a ``ConfigurationError`` if ``self.params`` is not empty.  We take ``class_name`` as
        an argument so that the error message gives some idea of where an error happened, if there
        was one.  ``class_name`` should be the name of the `calling` class, the one that got extra
        parameters (if there are any).
        """
        if self.args:
            raise ConfigurationError("Extra parameters passed to {}: {}".format(class_name, self.args))

    def __getitem__(self, key):
        if key in self.args:
            return self._check_is_dict(key, self.args[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.args[key] = value

    def __delitem__(self, key):
        del self.args[key]

    def __iter__(self):
        return iter(self.args)

    def __len__(self):
        return len(self.args)

    def _check_is_dict(self, new_history, value):
        if isinstance(value, dict):
            new_history = self.history + new_history + "."
            return Params(value,
                          history=new_history,
                          loading_from_archive=self.loading_from_archive,
                          files_to_archive=self.files_to_archive)
        if isinstance(value, list):
            value = [self._check_is_dict(new_history + '.list', v) for v in value]
        return value

    def set_defaults(self):
        """Make sure the commandline arguments are initialized properly."""
        # Check critical files exist
        self.args.pipeline.reader.model.encoding.batch_size = self.args.pipeline.data.dataProcessor.batch_size
        self.args.pipeline.reader.model.encoding.test_batch_size = self.args.pipeline.data.dataProcessor.test_batch_size
        self.args.pipeline.reader.model.aligning.doc_hidden_size = 2 * self.args.pipeline.reader.model.encoding.hidden_size
        self.args.pipeline.reader.model.aligning.question_hidden_size = 2 * self.args.pipeline.reader.model.encoding.hidden_size
        self.args.pipeline.data.dataProcessor.data_workers = self.args.runtime.data_workers

        self.args.files.dev_json = os.path.join(self.args.files.data_dir, self.args.files.dev_json)
        if not os.path.isfile(self.args.files.dev_json):
            raise IOError('No such file: %s' % self.args.files.dev_json)
        self.args.files.train_file = os.path.join(self.args.files.data_dir, self.args.files.train_file)
        if not os.path.isfile(self.args.files.train_file):
            raise IOError('No such file: %s' % self.args.files.train_file)
        self.args.files.dev_file = os.path.join(self.args.files.data_dir, self.args.files.dev_file)
        if not os.path.isfile(self.args.files.dev_file):
            raise IOError('No such file: %s' % self.args.files.dev_file)
        if self.args.files.embedding_file:
            self.args.files.embedding_file = os.path.join(self.args.files.embed_dir, self.args.files.embedding_file)
            if not os.path.isfile(self.args.files.embedding_file):
                raise IOError('No such file: %s' % self.args.files.embedding_file)

        # Set model directory
        subprocess.call(['mkdir', '-p', self.args.files.model_dir])

        # Set model name
        if not self.args.files.model_name:
            import uuid
            import time
            self.args.files.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

        # Set log + model file names
        self.args.files.log_file = os.path.join(self.args.files.model_dir, self.args.files.model_name + '.txt')
        self.args.files.model_file = os.path.join(self.args.files.model_dir, self.args.files.model_name + '.mdl')

        # Embeddings options
        if self.args.files.embedding_file:
            with open(self.args.files.embedding_file) as f:
                dim = len(f.readline().strip().split(' ')) - 1
            self.args.files.embedding_dim = dim
        elif not self.args.files.embedding_dim:
            raise RuntimeError('Either embedding_file or embedding_dim '
                               'needs to be specified.')

        # Make sure fix_embeddings and embedding_file are consistent
        if self.args.pipeline.reader.fix_embeddings:
            if not (self.args.files.embedding_file or self.args.files.pretrained):
                logger.warning('WARN: fix_embeddings set to False '
                               'as embeddings are random.')
                self.args.pipeline.reader.fix_embeddings = False


def pop_choice(params: Dict[str, Any],
               key: str,
               choices: List[Any],
               default_to_first_choice: bool = False,
               history: str = "?.") -> Any:
    """
    Performs the same function as :func:`Params.pop_choice`, but is required in order to deal with
    places that the Params object is not welcome, such as inside Keras layers.  See the docstring
    of that method for more detail on how this function works.

    This method adds a ``history`` parameter, in the off-chance that you know it, so that we can
    reproduce :func:`Params.pop_choice` exactly.  We default to using "?." if you don't know the
    history, so you'll have to fix that in the log if you want to actually recover the logged
    parameters.
    """
    value = Params(params, history).pop_choice(key, choices, default_to_first_choice)
    return value


def _replace_none(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    for key in dictionary.keys():
        if dictionary[key] == "None":
            dictionary[key] = None
        elif isinstance(dictionary[key], dict):
            dictionary[key] = _replace_none(dictionary[key])
    return dictionary


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


# below are all default configuration settings in DrQA

# class Param:
#     def __init__(self, parserName):
#         self.parser = argparse.ArgumentParser(
#             parserName,
#             formatter_class=argparse.ArgumentDefaultsHelpFormatter
#         )
#         self.args = None
#
#     def add_pipeline_args(self):
#         self.parser.register('type', 'bool', str2bool)
#
#         # Pipeline architecture
#         pipeline = self.parser.add_argument_group('DrQA Pipeline Architecture')
#         pipeline.add_argument('--reader-model', type=str, default=None,
#                             help='Path to trained Document Reader model')
#         pipeline.add_argument('--retriever-model', type=str, default=None,
#                             help='Path to Document Retriever model (tfidf)')
#         pipeline.add_argument('--doc-db', type=str, default=None,
#                             help='Path to Document DB')
#         pipeline.add_argument('--tokenizer', type=str, default=None,
#                             help=("String option specifying tokenizer type to "
#                                   "use (e.g. 'corenlp')"))
#         pipeline.add_argument('--candidate-file', type=str, default=None,
#                             help=("List of candidates to restrict predictions to, "
#                                   "one candidate per line"))
#
#     def add_model_args(self):
#         self.parser.register('type', 'bool', str2bool)
#
#         # Model architecture
#         model = self.parser.add_argument_group('DrQA Reader Model Architecture')
#         model.add_argument('--model-type', type=str, default='rnn',
#                            help='Model architecture type')
#         model.add_argument('--embedding-dim', type=int, default=300,
#                            help='Embedding size if embedding_file is not given')
#         model.add_argument('--hidden-size', type=int, default=128,
#                            help='Hidden size of RNN units')
#         model.add_argument('--doc-layers', type=int, default=3,
#                            help='Number of encoding layers for document')
#         model.add_argument('--question-layers', type=int, default=3,
#                            help='Number of encoding layers for question')
#         model.add_argument('--rnn-type', type=str, default='lstm',
#                            help='RNN type: LSTM, GRU, or RNN')
#
#         # Model specific details
#         detail = self.parser.add_argument_group('DrQA Reader Model Details')
#         detail.add_argument('--concat-rnn-layers', type='bool', default=True,
#                             help='Combine hidden states from each encoding layer')
#         detail.add_argument('--question-merge', type=str, default='self_attn',
#                             help='The way of computing the question representation')
#         detail.add_argument('--use-qemb', type='bool', default=True,
#                             help='Whether to use weighted question embeddings')
#         detail.add_argument('--use-in-question', type='bool', default=True,
#                             help='Whether to use in_question_* features')
#         detail.add_argument('--use-pos', type='bool', default=True,
#                             help='Whether to use pos features')
#         detail.add_argument('--use-ner', type='bool', default=True,
#                             help='Whether to use ner features')
#         detail.add_argument('--use-lemma', type='bool', default=True,
#                             help='Whether to use lemma features')
#         detail.add_argument('--use-tf', type='bool', default=True,
#                             help='Whether to use term frequency features')
#
#         # Optimization details
#         optim = self.parser.add_argument_group('DrQA Reader Optimization')
#         optim.add_argument('--dropout-emb', type=float, default=0.4,
#                            help='Dropout rate for word embeddings')
#         optim.add_argument('--dropout-rnn', type=float, default=0.4,
#                            help='Dropout rate for RNN states')
#         optim.add_argument('--dropout-rnn-output', type='bool', default=True,
#                            help='Whether to dropout the RNN output')
#         optim.add_argument('--optimizer', type=str, default='adamax',
#                            help='Optimizer: sgd or adamax')
#         optim.add_argument('--learning-rate', type=float, default=0.1,
#                            help='Learning rate for SGD only')
#         optim.add_argument('--grad-clipping', type=float, default=10,
#                            help='Gradient clipping')
#         optim.add_argument('--weight-decay', type=float, default=0,
#                            help='Weight decay factor')
#         optim.add_argument('--momentum', type=float, default=0,
#                            help='Momentum factor')
#         optim.add_argument('--fix-embeddings', type='bool', default=True,
#                            help='Keep word embeddings fixed (use pretrained)')
#         optim.add_argument('--tune-partial', type=int, default=0,
#                            help='Backprop through only the top N question words')
#         optim.add_argument('--rnn-padding', type='bool', default=False,
#                            help='Explicitly account for padding in RNN encoding')
#         optim.add_argument('--max-len', type=int, default=15,
#                            help='The max span allowed during decoding')
#
#     def add_train_args(self):
#         """
#         DONE
#         Adds commandline arguments pertaining to training a model. These
#         are different from the arguments dictating the model architecture.
#         """
#         self.parser.register('type', 'bool', str2bool)
#
#         # Runtime environment
#         runtime = self.parser.add_argument_group('Environment')
#         runtime.add_argument('--data-workers', type=int, default=5,
#                              help='Number of subprocesses for data loading')
#         runtime.add_argument('--random-seed', type=int, default=1013,
#                              help=('Random seed for all numpy/torch/cuda '
#                                    'operations (for reproducibility)'))
#         runtime.add_argument('--num-epochs', type=int, default=40,
#                              help='Train data iterations')
#         runtime.add_argument('--batch-size', type=int, default=32,
#                              help='Batch size for training')
#         runtime.add_argument('--test-batch-size', type=int, default=128,
#                              help='Batch size during validation/testing')
#
#         # Files
#         files = self.parser.add_argument_group('Filesystem')
#         files.add_argument('--model-dir', type=str, default=self.MODEL_DIR,
#                            help='Directory for saved models/checkpoints/logs')
#         files.add_argument('--model-name', type=str, default='',
#                            help='Unique model identifier (.mdl, .txt, .checkpoint)')
#         files.add_argument('--data-dir', type=str, default=self.DATA_DIR,
#                            help='Directory of training/validation data')
#         files.add_argument('--train-file', type=str,
#                            default='SQuAD-v1.1-train-processed-corenlp.txt',
#                            help='Preprocessed train file')
#         files.add_argument('--dev-file', type=str,
#                            default='SQuAD-v1.1-dev-processed-corenlp.txt',
#                            help='Preprocessed dev file')
#         files.add_argument('--dev-json', type=str, default='SQuAD-v1.1-dev.json',
#                            help=('Unprocessed dev file to run validation '
#                                  'while training on'))
#         files.add_argument('--embed-dir', type=str, default=self.EMBED_DIR,
#                            help='Directory of pre-trained embedding files')
#         files.add_argument('--embedding-file', type=str,
#                            default='glove.840B.300d.txt',
#                            help='Space-separated pretrained embeddings file')
#
#         # Saving + loading
#         save_load = self.parser.add_argument_group('Saving/Loading')
#         save_load.add_argument('--checkpoint', type='bool', default=False,
#                                help='Save model + optimizer state after each epoch')
#         save_load.add_argument('--pretrained', type=str, default='',
#                                help='Path to a pretrained model to warm-start with')
#         save_load.add_argument('--expand-dictionary', type='bool', default=False,
#                                help='Expand dictionary of pretrained model to ' +
#                                     'include training/dev words of new data')
#         # Data preprocessing
#         preprocess = parser.add_argument_group('Preprocessing')
#         preprocess.add_argument('--uncased-question', type='bool', default=False,
#                                 help='Question words will be lower-cased')
#         preprocess.add_argument('--uncased-doc', type='bool', default=False,
#                                 help='Document words will be lower-cased')
#         preprocess.add_argument('--restrict-vocab', type='bool', default=True,
#                                 help='Only use pre-trained words in embedding_file')
#
#         # General
#         general = self.parser.add_argument_group('General')
#         general.add_argument('--valid-metric', type=str, default='f1',
#                              help='The evaluation metric used for model selection')
#         general.add_argument('--display-iter', type=int, default=25,
#                              help='Log state after every <display_iter> epochs')
#
#     def add_predict_args(self):
#         self.parser.register('type', 'bool', str2bool)
#
#         # prediction args
#         parser = argparse.ArgumentParser()
#         parser.add_argument('dataset', type=str)
#         parser.add_argument('--out-dir', type=str, default='/tmp',
#                             help=("Directory to write prediction file to "
#                                   "(<dataset>-<model>-pipeline.preds)"))
#         parser.add_argument('--reader-model', type=str, default=None,
#                             help="Path to trained Document Reader model")
#         parser.add_argument('--retriever-model', type=str, default=None,
#                             help="Path to Document Retriever model (tfidf)")
#         parser.add_argument('--doc-db', type=str, default=None,
#                             help='Path to Document DB')
#         parser.add_argument('--embedding-file', type=str, default=None,
#                             help=("Expand dictionary to use all pretrained "
#                                   "embeddings in this file"))
#         parser.add_argument('--candidate-file', type=str, default=None,
#                             help=("List of candidates to restrict predictions to, "
#                                   "one candidate per line"))
#         parser.add_argument('--n-docs', type=int, default=5,
#                             help="Number of docs to retrieve per query")
#         parser.add_argument('--top-n', type=int, default=1,
#                             help="Number of predictions to make per query")
#         parser.add_argument('--tokenizer', type=str, default=None,
#                             help=("String option specifying tokenizer type to use "
#                                   "(e.g. 'corenlp')"))
#         parser.add_argument('--no-cuda', action='store_true',
#                             help="Use CPU only")
#         parser.add_argument('--gpu', type=int, default=-1,
#                             help="Specify GPU device id to use")
#         parser.add_argument('--parallel', action='store_true',
#                             help='Use data parallel (split across gpus)')
#         parser.add_argument('--num-workers', type=int, default=None,
#                             help='Number of CPU processes (for tokenizing, etc)')
#         parser.add_argument('--batch-size', type=int, default=128,
#                             help='Document paragraph batching size')
#         parser.add_argument('--predict-batch-size', type=int, default=1000,
#                             help='Question batching size')
#
