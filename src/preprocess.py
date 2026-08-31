from random import shuffle
import train
from utils import *;

# Loading the MINST Handwritten digits dataset
(train_images, train_labels), (_,_) = tf.keras.dataset.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28,28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the image to [-1,1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images),shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# *** Creating the Model

# GENERATOR
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias = False, input_shape = (100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7,7,256)))
    assert model.output_shape == (None, 7,7,256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128,(5,5), strides=(1,1), padding='same', use_bias= False))
    assert model.output_shape == (None, 7,7,128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakReLU)

    model.add(layers.Conv2DTranspose(1,(5,5), strides=(2,2), padding='same', use_bias=False, activation = 'tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model