INT   = 'int64'
FLOAT = 'float32'

from collections import namedtuple

# A simple type to hold parallel corpora samples
# src: List of source sentence words or None if not applicable
# trg: List of target sentence/description words
# split: A split index starting from 0 for multiple source/target pairs
# imgid: Index into the image feature matrix or None if not applicable
# imgname: Filename of the image or None if not applicable
Sample = namedtuple('Sample', ['src', 'trg', 'split', 'imgid', 'imgname'])
