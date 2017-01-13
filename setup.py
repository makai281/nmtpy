# -*- coding: utf-8 -*-
from setuptools import setup
import nmtpy

# Install pycocoevalcap metric scorers as well
pycocometrics  = ['bleu', 'meteor', 'cider', 'rouge']
pycocopackages = ['nmtpy.cocoeval.%s' % m for m in pycocometrics]

setup(
        name='nmtpy',
        version=nmtpy.__version__,
        description='NMT framework for Python based on Theano',
        url='http://github.com/lium-lst/nmtpy.git',
        author='Ozan Çağlayan',
        author_email='ozancag@gmail.com',
        license='MIT',
        packages=['nmtpy', 'nmtpy.models', 'nmtpy.iterators', 'nmtpy.metrics', 'nmtpy.cocoeval'] + pycocopackages,
        package_data={'' : ['data/*']}, # METEOR files
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
