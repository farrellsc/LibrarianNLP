> 20180729 on Core Model Development

# Objective

No error running `python -m unittest Aphorism_test.py` (tests.reader.models.Aphorism_test.py). Unittest document is [here](https://docs.python.org/3/library/unittest.html). You may also take tests.utils.DotDict_test.py as an example.

In order to do this you need to finish two classes: `LibNlp.reader.models.Aphorism` and `LibNlp.networks.BasicLSTM`. These two classes are subclasses of pytorch.nn.Module, so specifically you only need to implement the `forward` method.

Worth mentioning since dataProcess has not been finished, sadly you need to write your own data loading code and simple evaluations :-(. 

I think this [script](https://github.com/pytorch/examples/blob/master/mnist/main.py) would be of help.

# Development Usage

Everytime you modifies source code (in inner most LibNlp directory) you should run `python setup.py install` to rebuild so that new source code can take effect on testcases.
