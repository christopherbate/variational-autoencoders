import tensorflow as tf
import os
import time
import vae_model_v2
import matplotlib.pyplot as plt
from datetime import datetime

tf.keras.backend.clear_session()

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
train_images = tf.random.shuffle(train_images)

LOG_DIR = "logs/cvae/"+time.strftime("%Y%m%d-%H%M%S", time.localtime())
LOG_DIR_IMAGES = LOG_DIR + "/images"
TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000
EPOCHS = 200
LATENT_DIM = 50

# Create writers
image_log_writer = tf.summary.create_file_writer(LOG_DIR_IMAGES)

test_vector = tf.random.normal(shape=[16, LATENT_DIM])


# randomize
train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(
    test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)


# Setup training
stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
model = vae_model_v2.CVAE(latent_dim=LATENT_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
model.build((BATCH_SIZE, 28, 28, 1))
model.summary()


# Setup logging
img_step_idx = 0


def image_callback(batch, logs):
    if(batch % 1 == 0):
        predictions = tf.sigmoid(model.genNet(test_vector))
        with image_log_writer.as_default():
            tf.summary.image('Generated Images', predictions,
                             step=batch,
                             max_outputs=16)        


# Callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
    tf.keras.callbacks.LambdaCallback(on_epoch_end=image_callback)
]

# # Conduct Training
model.fit(train_images, train_images, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_data=(test_images, test_images),
          callbacks=callbacks)
