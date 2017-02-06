# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

from setuptools import setup
import nmtpy

# Install pycocoevalcap metric scorers as well
pycocometrics   = ['bleu', 'meteor', 'cider', 'rouge']
pycocopackages  = ['nmtpy.cocoeval.%s' % m for m in pycocometrics]

if 'install' in sys.argv or 'develop' in sys.argv:
    if not os.path.exists('nmtpy/external/data/paraphrase-en.gz'):
        print('You need to run scripts/get-meteor-data.sh first.')
        sys.exit(1)

setup(
        name='nmtpy',
        version=nmtpy.__version__,
        description='Neural Machine Translation Framework in Python',
        url='https://github.com/lium-lst/nmtpy',
        author='Ozan Çağlayan',
        author_email='ozancag@gmail.com',
        license='MIT',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2 :: Only',
            'Programming Language :: Python :: 2.7',
            'Operating System :: POSIX',
            ],
        keywords='nmt neural-mt translation deep-learning',
        packages=['nmtpy', 'nmtpy.models', 'nmtpy.iterators', 'nmtpy.metrics', 'nmtpy.cocoeval'] + pycocopackages,
        package_data={'' : ['external/meteor-1.5.jar', 'external/data/*gz', 'external/multi-bleu.perl']}, # data files
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
                    'bin/nmt-bpe-apply',
                    'bin/nmt-bpe-learn',
                ],
        zip_safe=False)
