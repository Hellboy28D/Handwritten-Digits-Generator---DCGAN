from utils import *;
from preprocess import *;


# Defining the training loop
EPOCH = 100
noise_dim = 100
num_example_to_generator = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF
seed = tf.random.normal([num_example_to_generator, noise_dim])

# Notice the use of 'tf.function'
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise =tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training = True)

        real_output = discriminator(images, training = True)
        fake_output = discriminator(generated_images, training = True)