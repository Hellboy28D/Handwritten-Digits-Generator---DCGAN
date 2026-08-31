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