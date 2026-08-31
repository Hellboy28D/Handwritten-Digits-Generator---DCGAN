from calendar import EPOCH
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


# Using the untrained generator to generate an image from random noise
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training = False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')


# Discriminator
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64,(5,5), strides=(2,2), padding = 'same',
                                    input_shape=[28,28,1]))
    
    model.add(layers.LeakReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128,(5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# Using the untrained discriminator to predict whether an imagge is real or fake
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)


# *** Loss and Optimizer ***
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
tf.ones_like((1,1,1,1,1,1,1,1,1,0,0))

# Dicrimnator Loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.one_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Generator Loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Saving the checkpoints

checkpoint_dir = ''
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf. train.Checkpoint(generator_optimizer = generator_optimizer,
                                  discriminator_optimizer = discriminator_optimizer,
                                  generator = generator,
                                  discriminator = discriminator)


# Defining the training loop
EPOCH = 100
noise_dim = 100
num_example_to_generator = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF
seed = tf.random.normal([num_example_to_generator, noise_dim])

# Notice the use of 'tf.function'