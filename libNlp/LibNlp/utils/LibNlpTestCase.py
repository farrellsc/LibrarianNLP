# pylint: disable=invalid-name,protected-access
import logging
import os
import shutil
from unittest import TestCase
from pathlib import PosixPath

os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'data')


class LibNlpTestCase(TestCase):  # pylint: disable=too-many-public-methods
    """
    A custom subclass of :class:`~unittest.TestCase` that disables some of the
    more verbose AllenNLP logging and that creates and destroys a temp directory
    as a test fixture.
    """
    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.DEBUG)
