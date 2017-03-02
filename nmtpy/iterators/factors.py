#!/usr/bin/env python
from six.moves import range
from six.moves import zip

import numpy as np

from collections import OrderedDict

from ..sysutils import fopen
from .iterator  import Iterator
from .homogeneous import HomogeneousData
from ..nmtutils import sent_to_idx
#from ..typedef import *

"""Parallel text iterator for translation data."""
class FactorsIterator(Iterator):
    def __init__(self, batch_size, seed=1234, mask=True, shuffle_mode=None, **kwargs):
        super(FactorsIterator, self).__init__(batch_size, seed, mask, shuffle_mode)

#        assert 'srcfile' in kwargs, "Missing argument srcfile"
#        assert 'trglemfile' in kwargs, "Missing argument trglemfile"
#        assert 'trgfactfile' in kwargs, "Missing argument trgfactfile"
#        assert 'srcdict' in kwargs, "Missing argument srcdict"
#        assert 'trglemdict' in kwargs, "Missing argument trglemdict"
#        assert 'trgfactdict' in kwargs, "Missing argument trgfactdict"
        
	#TODO add pkl files reading

	# How do we use the multimodal data?
        # 'all'     : All combinations (~725K parallel)
        # 'single'  : Take only the first pair e.g., train0.en->train0.de (~29K parallel)
        # 'pairs'   : Take only one-to-one pairs e.g., train_i.en->train_i.de (~145K parallel)
        self.mode = kwargs.get('mode', 'all')

	if 'srcfactfile' in kwargs:
	    self.srcfact = True
	    self.trgfact = True
	    self.srcfile = kwargs['srcfile']
	    self.srcfactfile = kwargs['srcfactfile']

	    self.srcdict = kwargs['srcdict']
	    self.srcfactdict = kwargs['srcfactdict']
	    
	    self.n_words_src = kwargs.get('n_words_src', 0)
	    self.n_words_srcfact = kwargs.get('n_words_srcfact', 0)
	    
	    self.src_name = kwargs.get('src_name', 'x1')
	    self.srcfact_name = kwargs.get('srcfact_name', 'x2')
	    self._keys = [self.src_name]
	    self._keys.append(self.srcfact_name)
	    if self.mask:
		self._keys.append("%s_mask" % self.src_name)
		self._keys.append("%s_mask" % self.srcfact_name)
	else:
	    self.srcfact = False
	    self.srcfile = kwargs['srcfile']
	    self.srcdict = kwargs['srcdict']
	    self.n_words_src = kwargs.get('n_words_src', 0)
	    self.src_name = kwargs.get('src_name', 'x')
	    self._keys = [self.src_name]
	    if self.mask:
		self._keys.append("%s_mask" % self.src_name)

	if 'trgfactfile' in kwargs:
	    self.trgfact = True
	    self.trglemfile = kwargs['trglemfile']
	    self.trgfactfile = kwargs['trgfactfile']

	    self.trglemdict = kwargs['trglemdict']
	    self.trgfactdict = kwargs['trgfactdict']

	    self.n_words_trglem = kwargs.get('n_words_trglem', 0)
	    self.n_words_trgfact = kwargs.get('n_words_trgfact', 0)

	    self.trglem_name = kwargs.get('trglem_name', 'y1')
	    self.trgfact_name = kwargs.get('trgfact_name', 'y2')

	    self._keys.append(self.trglem_name)
	    self._keys.append(self.trgfact_name)
	    if self.mask:
		self._keys.append("%s_mask" % self.trglem_name)
		self._keys.append("%s_mask" % self.trgfact_name)
	
	else:
	    self.trgfact = False
	    self.trgfile = kwargs['trgfile']
	    self.trgdict = kwargs['trgdict']
	    self.n_words_trg = kwargs.get('n_words_trg', 0)
	    self.trg_name = kwargs.get('trg_name', 'y')
	    self._keys.append(self.trg_name)
	    if self.mask:
		self._keys.append("%s_mask" % self.trg_name)

    def read(self):
        seqs = []
	if self.srcfact and self.trgfact:
	    tlf = fopen(self.trglemfile, 'r')
	    tff = fopen(self.trgfactfile, 'r')
	    slf = fopen(self.srcfile, 'r')
	    sff = fopen(self.srcfactfile, 'r')

	    for idx, (slline, sfline, tlline, tfline) in enumerate(zip(slf, sff, tlf, tff)):
		slline = slline.strip()
		sfline = sfline.strip()
		tlline = tlline.strip()
		tfline = tfline.strip()
		
		# Exception if empty line found
		if slline == "" or sfline == "" or tlline == "" or tfline == "":
		    continue
            
		slseq = [self.srcdict.get(w, 1) for w in slline.split(' ')]
		sfseq = [self.srcfactdict.get(w, 1) for w in sfline.split(' ')]
		tlseq = [self.trglemdict.get(w, 1) for w in tlline.split(' ')]
		tfseq = [self.trgfactdict.get(w, 1) for w in tfline.split(' ')]
            
		# if given limit vocabulary
		if self.n_words_src > 0:
		    slseq = [w if w < self.n_words_src else 1 for w in slseq]
		if self.n_words_srcfact > 0:
		    sfseq = [w if w < self.n_words_srcfact else 1 for w in sfseq]
		if self.n_words_trglem > 0:
		    tlseq = [w if w < self.n_words_trglem else 1 for w in tlseq]
		if self.n_words_trgfact > 0:
		    tfseq = [w if w < self.n_words_trgfact else 1 for w in tfseq]
	    
		# Append sequences to the list
		seqs.append((slseq, sfseq, tlseq, tfseq))

	    slf.close()
	    sff.close()
	    tlf.close()
	    tff.close()

	elif self.srcfact:
	    slf = fopen(self.srcfile, 'r')
	    sff = fopen(self.srcfactfile, 'r')
	    tf = fopen(self.trgfile, 'r')

	    for idx, (slline, sfline, tline) in enumerate(zip(slf, sff, tf)):
		slline = slline.strip()
		sfline = sfline.strip()
		tline = tline.strip()
		
		# Exception if empty line found
		if slline == "" or sfline == "" or tline == "":
		    continue
            
		slseq = [self.srcdict.get(w, 1) for w in slline.split(' ')]
		sfseq = [self.srcfactdict.get(w, 1) for w in sfline.split(' ')]
		tseq = [self.trgdict.get(w, 1) for w in tline.split(' ')]
            
		# if given limit vocabulary
		if self.n_words_src > 0:
		    slseq = [w if w < self.n_words_src else 1 for w in slseq]
		if self.n_words_srcfact > 0:
		    sfseq = [w if w < self.n_words_srcfact else 1 for w in sfseq]
		if self.n_words_trg > 0:
		    tfseq = [w if w < self.n_words_trg else 1 for w in tseq]
	    
		# Append sequences to the list
		seqs.append((slseq, sfseq, tseq))

	    slf.close()
	    sff.close()
	    tf.close()

	elif self.trgfact:
	    # We open the data files
	    sf = fopen(self.srcfile, 'r')
	    tlf = fopen(self.trglemfile, 'r')
	    tff = fopen(self.trgfactfile, 'r')
	    # We iterate the data files
	    for idx, (sline, tlline, tfline) in enumerate(zip(sf, tlf, tff)):
		sline = sline.strip()
		tlline = tlline.strip()
		tfline = tfline.strip()

		# Exception if empty line found
		if sline == "" or tlline == "" or tfline == "":
		    continue
		# For each word in the sentence we add its corresponding ID in the dic
	    	seq = [self.srcdict.get(w, 1) for w in sline.split(' ')]
	    	lseq = [self.trglemdict.get(w, 1) for w in tlline.split(' ')]
	    	fseq = [self.trgfactdict.get(w, 1) for w in tfline.split(' ')]

	    	#if given limit vocabulary
		if self.n_words_src > 0:
		    sseq = [w if w < self.n_words_src else 1 for w in seq]

		# if given limit vocabulary
		if self.n_words_trglem > 0:
		    tlseq = [w if w < self.n_words_trglem else 1 for w in lseq]
		if self.n_words_trgfact > 0:
		    tfseq = [w if w < self.n_words_trgfact else 1 for w in fseq]

		# Append sequences to the list
		seqs.append((sseq, tlseq, tfseq))
        
	    sf.close()
	    tlf.close()
	    tff.close()

        # Save sequences
        self._seqs = seqs

        # Number of training samples
        self.n_samples = len(self._seqs)
        # Some statistics
#        unk_trg = 0
#        unk_src = 0
#        total_src_words = []
#        total_trg_words = []

        # Let's map the sentences once to idx's
#	if self.srcfact:
#	    for sample in self._seqs:
#		sample[0] = sent_to_idx(self.srclem_dict, sample[0], self.n_words_srclem)
#		total_srclem_words.extend(sample[0])
#		sample[1] = sent_to_idx(self.srcfact_dict, sample[1], self.n_words_srcfact)
#		total_srcfact_words.extend(sample[1])
#		sample[2] = sent_to_idx(self.trglem_dict, sample[2], self.n_words_trglem)
#		total_trglem_words.extend(sample[2])
#		sample[3] = sent_to_idx(self.trgfact_dict, sample[3], self.n_words_trgfact)
#		total_trg_words.extend(sample[3])
#	else:
#	    for sample in self._seqs:
#		sample[0] = sent_to_idx(self.srcdict, sample[0], self.n_words_src)
#		total_src_words.extend(sample[0])
#		sample[1] = sent_to_idx(self.trglemdict, sample[1], self.n_words_trglem)
#		total_trglem_words.extend(sample[1])
#		sample[2] = sent_to_idx(self.trgfactdict, sample[2], self.n_words_trgfact)
#		total_trgfact_words.extend(sample[2])
        
#        self.unk_src = total_src_words.count(1)
#        self.unk_trg = total_trg_words.count(1)
#        self.total_src_words = len(total_src_words)
#        self.total_trg_words = len(total_trg_words)


        if self.shuffle_mode == 'trglen':
            # Homogeneous batches ordered by target sequence length
            # Get an iterator over sample idxs
            self._iter = HomogeneousData(self._seqs, self.batch_size, trg_pos=1)
            self._process_batch = (lambda idxs: self.mask_seqs(idxs))
        else:
            if self.shuffle_mode == 'simple':
                # Simple shuffle
                self._idxs = np.random.permutation(self.n_samples)
            else:
                # Ordered
                self._idxs = np.arange(self.n_samples)
            self.prepare_batches()


    @staticmethod
    def mask_data_mult(seqs):
	"""Pads sequences with EOS (0) for minibatch processing."""
	lengths = [len(s) for s in seqs]
	n_samples = len(seqs)
	 
	maxlen = np.max(lengths) + 1
	 
	# Shape is (t_steps, samples)
	x = np.zeros((maxlen, n_samples)).astype(INT)
	x_mask = np.zeros_like(x).astype(FLOAT)
	 
	for idx, s_x in enumerate(seqs):

	    x[:lengths[idx], idx] = s_x
	    x_mask[:lengths[idx], idx] = 1.
	    
	return x, x_mask


    def mask_seqs(self, idxs):
        """Prepares a list of padded tensors with their masks for the given sample idxs."""
	if self.srcfact and self.trgfact:
	    src, src_mask = Iterator.mask_data([self._seqs[i][0] for i in idxs])
	    srcfact, srcmult_mask = Iterator.mask_data([self._seqs[i][1] for i in idxs])
	    trg, trg_mask = Iterator.mask_data([self._seqs[i][2] for i in idxs])
	    trgmult, trgmult_mask = Iterator.mask_data([self._seqs[i][3] for i in idxs])
	    return (src, srcfact, src_mask, trg, trgmult, trg_mask)
	elif self.srcfact:
	    src, src_mask = Iterator.mask_data([self._seqs[i][0] for i in idxs])
	    srcfact, srcmult_mask = Iterator.mask_data([self._seqs[i][1] for i in idxs])
	    trg, trg_mask = Iterator.mask_data([self._seqs[i][2] for i in idxs])
	    return (src, srcfact, src_mask, srcmult_mask, trg, trg_mask)
	elif self.trgfact:
	    src, src_mask = Iterator.mask_data([self._seqs[i][0] for i in idxs])
	    trg, trg_mask = Iterator.mask_data([self._seqs[i][1] for i in idxs])
	    trgmult, trgmult_mask = self.mask_data_mult([self._seqs[i][2] for i in idxs])
	    return (src, src_mask, trg, trgmult, trg_mask, trgmult_mask)

    def prepare_batches(self):
        self._minibatches = []

        for i in range(0, self.n_samples, self.batch_size):
            batch_idxs = self._idxs[i:i + self.batch_size]
            self._minibatches.append(self.mask_seqs(batch_idxs))

        self.rewind()

    def rewind(self):
        if self.shuffle_mode != 'trglen':
            self._iter = iter(self._minibatches)
