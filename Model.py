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

numSamples = len(data)

# The maximum length sentence we want for a single input in characters
seq_length = 280

a = dataEmbed(data, seq_length)
a.embed()
vocabSize = a.getVocabLen()
bData = a.getEmbed()

print(bData.shape)

inputData = bData[:, :279]
targetData = bData[:, 1:]

# print(inputData[0], "     ", targetData[0])

print("Input data shape: ", str(inputData.shape))
print("Target data shape: ", str(targetData.shape))

inputD = to_categorical(inputData, num_classes=vocabSize)
targetD = to_categorical(targetData, num_classes=vocabSize)
finalData = to_categorical(bData, num_classes=vocabSize)

print(inputD.shape)
print(targetD.shape)
print(finalData.shape)

embeddingDim = 256


#examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
#char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

#for i in char_dataset.take(5):
#  print(idx2char[i.numpy()])

#has not been tested, will test when other parts of code are working (can input data)
def getModel(vocab_size, embedding_dim, rnn_units, batch_size):
	model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)])
	# model.summary()
	return model

model = getModel(vocabSize, embeddingDim, 1024, numSamples)

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# example_batch_loss  = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(bData, epochs=10, callbacks=[checkpoint_callback])