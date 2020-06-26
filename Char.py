'''Ada Toydemir | 6/21/2020 '''
import numpy as np
import tensorflow as tf
'''
This assumes that data has already been padded so that all the
elements of the textArray are the same length. (which is nLen)

must call embed before getting the embeddings...
'''
class dataEmbed:
	'''
	' _nLen 		= number of characters per tweet
	' _result 		= np array of all embedded tweets
	' _textArray 	= input (list of all the tweets)
	' _char2idx		= dictionary of char key idx value
	' _idx2char		= dictionary of idx key char value
	' _vocabLen		= size of vocab
	'''

	def __init__(self, textArray, nLen):
		self._nLen = nLen
		self._result = np.empty((0, nLen))
		self._textArray = textArray
		wSpace = ''
		longForm = wSpace.join(textArray)
		self._char2idx, self._idx2char, self._vocabLen = self.charEmbedding(longForm)

	#function takes one big string
	def charEmbedding(self, text):
		#get the vocabulary space from 
		vocab = sorted(set(text))
		print ('{} unique characters'.format(len(vocab)))
		char2idx = {u:i for i, u in enumerate(vocab)}
		idx2char = np.array(vocab)
		return char2idx, idx2char, len(vocab)

	def embed(self):
		print(len(self._textArray))
		for text in self._textArray:
			textInt = np.zeros((1, self._nLen))
			for i in range(len(text)):
				textInt[0][i] = self._char2idx[text[i]]
			self._result = np.vstack((self._result, textInt[0]))

	def getVocabLen(self):
		return self._vocabLen

	def getEmbed(self):
		return self._result