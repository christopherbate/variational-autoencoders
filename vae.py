import tensorflow as tf
import os
import time
import numpy as numpy
import matplotlib.pyplot as plt
import PIL
import imageio
import vae_model

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(
    train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(
    test_images.shape[0], 28, 28, 1).astype('float32')

train_images /= 255.0
test_images /= 255.0

# Bin the images
train_images[train_images >= .5] = 1.0
train_images[train_images <= .5] = 0.0
test_images[test_images >= .5] = 1.0
test_images[test_images < 0.5] - 0.0

TRAIN_BUF = 60000
BATCH_SIZE = 10
TEST_BUF = 10000
EPOCHS = 5

# randomize
train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

# Setup training
model = vae_model.CVAE(10)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
test_vector = tf.random.normal(shape=[16, 10])


def visualize(model, z_test):
    # Visualize what we are learning.
    predictions = tf.sigmoid(model.generative_net(z_test))
    for i in range(test_vector.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0])
    plt.draw()
    plt.pause(0.001)


train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        loss = vae_model.compute_loss(model, image)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

# vae_model.compute_loss(model, train_images[0, :, :, 0])
for epoch in range(EPOCHS):
    start_time = time.time()
    # Training
    for idx, train_batch in enumerate(train_dataset):
        train_step(train_batch)
        if(idx % 100 == 0):
            print("Epoch {}, Batch {}, Loss {}".format(
                epoch, idx, train_loss.result()))
            visualize(model, test_vector)

    end_time = time.time()

    # Print stats
    print("Epoch: {}, time {}".format(epoch, end_time-start_time))

plt.show()
