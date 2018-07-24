from setuptools import setup, find_packages
import codecs


def read_install_requirements():
    with codecs.open('./requirements.txt', 'r', encoding='utf8') as f:
        content = f.readlines()
        requirements = [line.strip('\n') for line in content]
    return requirements


setup(
    name="LibrarianNLP",
    version="0.0",
    description='',
    keywords='NLP',
    url='https://github.com/farrellsc/LibrarianNLP',
    author='Zhou Zhuang',
    license='MIT',
    packages=find_packages(),
    install_requires=read_install_requirements(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    include_package_data=True,
    python_requires='>=3.5'
)
