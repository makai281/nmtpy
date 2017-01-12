# -*- coding: utf-8 -*-
from setuptools import setup
import nmtpy

setup(
        name='nmtpy',
        version=nmtpy.__version__,
        description='NMT framework for Python based on Theano',
        url='http://github.com/lium-lst/nmtpy.git',
        author='Ozan Çağlayan',
        author_email='ozancag@gmail.com',
        license='MIT',
        packages=['nmtpy', 'nmtpy.models', 'nmtpy.iterators', 'nmtpy.metrics', 'external.pycocoevalcap'],
        data_files=[('share/nmtpy/meteor', ['external/pycocoevalcap/meteor/meteor-1.5.jar'])],
        install_requires=[
          'numpy',
          'theano',
          'six',
        ],
        scripts=[
                    'bin/nmt-train',
                    'bin/nmt-extract',
                    'bin/nmt-translate',
                    'bin/nmt-build-dict',
                    'bin/nmt-coco-metrics',
                    'external/subword-nmt/nmt-bpe-apply',
                    'external/subword-nmt/nmt-bpe-learn',
                ],
        zip_safe=False)
