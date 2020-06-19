import tensorflow as tf
import numpy as np
from tf.keras.layers import Embedding, GRU, Dense
from tf import keras 

embeddingDim = 128

#has not been tested, will test when other parts of code are working (can input data)
def getModel(dataLen, vocabSize, batchSize, rnnUnits):
	inputs = keras.Input(shape=(dataLen,))
	x = Embedding(vocabSize, embeddingDim, batch_input_shape=[batchSize, none])(inputs)
	x = GRU(rnnUnits, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')(x)
	xOUT = Dense(vocabSize)(x)
	model = keras.Model(inputs, xOUT)
	return model