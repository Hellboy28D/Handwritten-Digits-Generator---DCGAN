from utils import *;
from preprocess import *;
from train import *;


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

    for image_batch in dataset:
        train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait =True)
    generated_and_save_images(generator,
                              epoch + 1,
                              seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Time for epoch {} is {} sec'. format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait = True)
    generated_and_save_images(generator,
                              epochs,
                              seed)


# Generate and save images

def generate_and_save(model, epoch, test_input):

    # Notices 'training' is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training = False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap = 'gray')
        plt.axis('off')
    
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# *** Training the model ***

