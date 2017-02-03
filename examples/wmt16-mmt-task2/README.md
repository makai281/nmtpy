# WMT16 Shared task on Multimodal Translation
## Task 2 - Cross-lingual Image Description Generation

### Multi30k Dataset

A copy of the original text files for Task 2 are available under `data/`. These files are downloaded
from [WMT16 Multimodal Task](http://www.statmt.org/wmt16/multimodal-task.html) webpage.
 
### Normalization and Tokenization

Make sure that the following scripts from the `mosesdecoder` project are in your `$PATH`:
  - tokenizer.perl
  - normalize-punctuation.perl

Run `scripts/01-tokenize.sh ~/nmtpy/data/wmt16-task2` to:
  - Normalize punctuations
  - Tokenize

train, val and test files from `data/` and save them under `~/nmtpy/data/wmt16-task2`.
**Note that** the output folder is in accordance with the configuration file
`wmt16-task2-monomodal.conf` so if you use another output folder, change the configuration
file as well.

### Preparing Data

`scripts/02-prepare.py` is a Python script that consumes all the tokenized data produces:
  - processed text files
  - `pkl` files to be used by `WMTIterator`
  - `nmtpy` dictionary files for source and target vocabularies
  
You can run the following command to prepare above files:
```
ODIR=~/nmtpy/data/wmt16-task2
scripts/02-prepare.py -i data/split_all.txt \
                -t $ODIR/train.*.en  -T $ODIR/train.*.de \
                -v $ODIR/val.*.en    -V $ODIR/val.*.de \
                -e $ODIR/test.*.en   -E $ODIR/test.*.de \
                -l -s -d 5 -o $ODIR # lowercase, strippunct, minwordoccurrence5
```
  
The produced `.pkl` data files contain a `list` of samples for each of the train/val/test sets
where a sample is represented with:
 - `ssplit`: An integer between 0-4 representing from which file the source sentence came from
 - `tsplit`: An integer between 0-4 representing from which file the source sentence came from
 - `imgid`: An integer between 0-(N-1) representing the order of the image for a set containing N images
 - `imgname`: The name of the JPG image file
 - `swords`: List of source words
 - `twords`: List of target words
 
Let's see with a concrete example:
```bash
cd $ODIR
ipython
```

```python
...
In [1]: import cPickle

In [2]: v = cPickle.load(open('flickr_30k_align.valid.pkl'))

In [3]: len(v)
Out[3]: 25350

In [4]: v[0]
Out[4]: 
[0,
 0,
 0,
 '1018148011.jpg',
 [u'a',
  u'group',
  u'of',
  u'people',
  u'stand',
  u'in',
  u'the',
  u'back',
  u'of',
  u'a',
  u'truck',
  u'filled',
  u'with',
  u'cotton'],
 [u'baumwolllager', u'mit', u'lkw']]
```

A clarification should be made about the number of samples in a set: since we have 5 source and 5 target sentences for each image, the script generates `5x5=25` comparable pairs for a single image. Since the validation set contains 1014 images, this makes a total of `25*1014=25350` samples.

During training, you can select whether you would like to use:
 - All 25 comparable pairs for an image (`data_mode:all`)
 - 5 comparable pairs for an image (`data_mode:pairs`)
   - `(.1.en, .1.de), (.2.en, .2.de), ..., (.5.en, .5.de)`
 - Just one pair from the first pair of files: `.1.en -> .1.de` (`data_mode:single`)
 
During early-stopping, we use by default `single` for validation to only consider the description pairs from `.1.en, .1.de` resulting in 1014 images-captions.
