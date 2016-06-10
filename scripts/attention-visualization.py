#!/usr/bin/env python

import cPickle as pkl
import numpy

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import skimage
import skimage.transform
import skimage.io

from PIL import Image

# keep aspect ratio, and center crop
def load_image(file_name, resize=256, crop=224):
  image = Image.open(file_name)
  width, height = image.size

  if width > height:
    width = (width * resize) / height
    height = resize
  else:
    height = (height * resize) / width
    width = resize
  left = (width  - crop) / 2
  top  = (height - crop) / 2
  image_resized = image.resize((width, height), Image.BICUBIC).crop((left, top, left + crop, top + crop))
  data = numpy.array(image_resized.convert('RGB').getdata()).reshape(crop, crop, 3)
  data = data.astype('float32') / 255
  return data

# TODO: Load the model for inference
# TODO: Load the images, maybe select 20 random images or so?
# TODO: Build everything necessary for build_sampler (e.g. nmt-translate)
#       We need f_init, f_next and f_alpha(?)

# build the sampling functions and model
trng = RandomStreams(1234)
use_noise = theano.shared(numpy.float32(0.), name='use_noise')

params = capgen.init_params(options)
params = capgen.load_params(model, params)
tparams = capgen.init_tparams(params)

# word index
f_init, f_next = capgen.build_sampler(tparams, options, use_noise, trng)

##############################
idx = numpy.random.randint(0, len(valid[0])) # random image
k = 1 # beam width
use_gt = False # set to False if you want to use the generated sample
gt = valid[0][idx][0] # groundtruth
context = numpy.array(valid[1][valid[0][idx][1]].todense()).reshape([14*14, 512]) # annotations
img = LoadImage(image_path+flist[valid[0][idx][1]])

if not use_gt:
    sample, score = capgen.gen_sample(tparams, f_init, f_next, context, 
                                      options, trng=trng, k=k, maxlen=200, stochastic=False)
    sidx = numpy.argmin(score)
    caption = sample[sidx][:-1]

# print the generated caption and the ground truth
if use_gt:
    caption = map(lambda w: worddict[w] if worddict[w] < options['n_words'] else 1, gt.split())
words = map(lambda w: word_idict[w] if w in word_idict else '<UNK>', caption)
print 'Sample:', ' '.join(words)
print 'GT:', gt

alpha = f_alpha(numpy.array(caption).reshape(len(caption),1), 
                numpy.ones((len(caption),1), dtype='float32'), 
                context.reshape(1,context.shape[0],context.shape[1]))
if options['selector']:
    sels = f_sels(numpy.array(caption).reshape(len(caption),1), 
                   numpy.ones((len(caption),1), dtype='float32'), 
                   context.reshape(1,context.shape[0],context.shape[1]))

# display the visualization
n_words = alpha.shape[0] + 1
w = numpy.round(numpy.sqrt(n_words))
h = numpy.ceil(numpy.float32(n_words) / w)
        
plt.subplot(w, h, 1)
plt.imshow(img)
plt.axis('off')

smooth = True

for ii in xrange(alpha.shape[0]):
    plt.subplot(w, h, ii+2)
    lab = words[ii]
    if options['selector']:
        lab += '(%0.2f)'%sels[ii]
    plt.text(0, 1, lab, backgroundcolor='white', fontsize=13)
    plt.text(0, 1, lab, color='black', fontsize=13)
    plt.imshow(img)
    if smooth:
        alpha_img = skimage.transform.pyramid_expand(alpha[ii,0,:].reshape(14,14), upscale=16, sigma=20)
    else:
        alpha_img = skimage.transform.resize(alpha[ii,0,:].reshape(14,14), [img.shape[0], img.shape[1]])
    plt.imshow(alpha_img, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
plt.show()
