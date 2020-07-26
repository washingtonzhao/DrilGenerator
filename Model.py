import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras import Sequential
from tensorflow import keras 
from tensorflow.keras.utils import to_categorical
import os

from Char import dataEmbed
from tweets import readJson, readJsonBlock

fileLocation = "data.json"
data = readJsonBlock(fileLocation)

#temporary limitation of amount of data being used
data = data[:280*64]

numSamples = len(data)

# The maximum length sentence we want for a single input in characters
seq_length = 279

a = dataEmbed(data, seq_length)
a.embed()
idx2char = a.getIDX2CHAR()
vocabSize = a.getVocabLen()
bData = a.getEmbed()

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(bData)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
	input_text = chunk[:-1]
	target_text = chunk[1:]
	return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = int((numSamples/280)/4)
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

embeddingDim = 256

def getModel(vocab_size, embedding_dim, rnn_units, batch_size):
	model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
	tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
	tf.keras.layers.Dense(vocab_size)])
	# model.summary()
	return model

model = getModel(vocabSize, embeddingDim, 1024, BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
	example_batch_predictions = model(input_example_batch)

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(bData, batch_size=BATCH_SIZE, epochs=10, callbacks=[checkpoint_callback])