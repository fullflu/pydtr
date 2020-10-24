#!/usr/bin/env python
# coding: utf-8

import os
import sys

from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install
from codecs import open

from src.pydtr.version import __version__

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != __version__:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, __version__
            )
            sys.exit(info)


setup(
    name='pydtr',
    version=__version__,
    description='Python library of Dynamic Treatment Regimes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fullflu/pydtr',
    author='fullflu',
    author_email='k.takayama0902@gmail.com',
    license='BSD',
    install_requires=[
        'pandas>=1.1.2',
        'scikit-learn>=0.23.2',
        'numpy>=1.19.2',
        'statsmodels>=0.12.0'
    ],
    keywords=['dynamic treatment regimes', 'reinforcement learning', 'dtr'],
    include_package_data=True,
    package_dir={'': "src"},
    packages=find_packages('src'),
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov", "coverage", "category_encoders"],
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
