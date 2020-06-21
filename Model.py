import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras import Sequential
from tensorflow import keras 

from Char import dataEmbed

embeddingDim = 256

### MIGHT NEED TO DO ONE-HOT ENCODING??? ###
### website says # (batch_size, sequence_length, vocab_size) ###

# The maximum length sentence we want for a single input in characters
seq_length = 280
#examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
#char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

#for i in char_dataset.take(5):
#  print(idx2char[i.numpy()])

#need to expand_dims or create batches (shape is batch x data x 280 (max tweet size))

#has not been tested, will test when other parts of code are working (can input data)
def getModel(vocab_size, embedding_dim, rnn_units, batch_size):
	model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)])
	model.summary()
	return model

getModel(26, embeddingDim, 1024, 9999)