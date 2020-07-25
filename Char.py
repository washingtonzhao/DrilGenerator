'''
Author: Ada Toydemir
Created: 6/21/2020
Most Recently Updated: 7/25/20'''

import numpy as np
import tensorflow as tf
'''
This assumes that data has already been padded, Data is inputted as one long
text. Must call embed before getting the embeddings...
'''
class dataEmbed:
	'''
	' _nLen 		= number of characters per tweet
	' _result 		= list of all embedded characters
	' _text 		= input (list of all the characters)
	' _char2idx		= dictionary of char key idx value
	' _idx2char		= dictionary of idx key char value
	' _vocabLen		= size of vocab
	'''

	def __init__(self, text, nLen):
		self._nLen = nLen
		self._result = []
		self._text = text
		self._char2idx, self._idx2char, self._vocabLen = self.charEmbedding(text)
		print("Iinitialization is complete.")

	#function takes one big string
	def charEmbedding(self, text):
		#get the vocabulary space from 
		vocab = sorted(set(text))
		print ('{} unique characters'.format(len(vocab)))
		char2idx = {u:i for i, u in enumerate(vocab)}
		idx2char = np.array(vocab)
		print("Char embeddings have been created.")
		return char2idx, idx2char, len(vocab)

	def embed(self):
		print(len(self._text))
		for char in self._text:
			self._result.append(self._char2idx[char])
		print('{} ---- characters mapped to int ---- > {}'.format(repr(self._text[:13]), self._result[:13]))
		print("Text has been embedded.")

	def getVocabLen(self):
		return self._vocabLen

	def getEmbed(self):
		return self._result

	def getIDX2CHAR(self):
		return self._idx2char