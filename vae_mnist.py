import tensorflow as tf
import os
import time
import vae_model
from datetime import datetime

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
EPOCHS = 50
LATENT_DIM = 50

# randomize
train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)


# Setup training
stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
mode_name = "CVAE_MNIST"
model = vae_model.CVAE(LATENT_DIM)
optimizer = tf.keras.optimizers.Adam()
test_vector = tf.random.normal(shape=[10, LATENT_DIM])


# Setup logging
train_summary_writer = tf.summary.create_file_writer(
    'logs/'+mode_name+"/"+stamp+"/train")
test_summary_writer = tf.summary.create_file_writer(
    'logs/'+mode_name+"/"+stamp+"/test")
graph_summary_writer = tf.summary.create_file_writer(
    "logs/"+mode_name+"/"+stamp
)
profile_logdir = 'logs/'+mode_name+"/"+stamp

# Logging - Metrics
train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()

# Define training
@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        loss = vae_model.compute_loss(model, image)
    gradients = tape. gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)


# Quick profile and trace
tf.summary.trace_on(graph=True, profiler=True)
graph_data = train_dataset.take(BATCH_SIZE)
first_batch = None
for batch in graph_data:
    train_step(batch)
    first_batch = batch
    break
with graph_summary_writer.as_default():
    tf.summary.trace_export(
        name="my_func_trace",
        step=0,
        profiler_outdir=profile_logdir)


# Define logging for tensorboard
def board_logging(model, z_test, step):    
    predictions = tf.sigmoid(model.generative_net(z_test))
    tf.summary.image("Generated Image Samples ",
                     predictions, step=step, max_outputs=10)
    tf.summary.scalar('Training Loss', train_loss.result(),
                      step=optimizer.iterations)
    train_loss.reset_states()

    conv1_weights = model.weights[0].numpy().reshape((-1, 3, 3, 1))
    tf.summary.image("CONV1", conv1_weights, step=step, max_outputs=10)
    conv2_weights = model.weights[2].numpy().reshape((-1, 3, 3, 1))
    tf.summary.image("CONV2", conv2_weights, step=step, max_outputs=10)


# Conduct Training
for epoch in range(EPOCHS):
    start_time = time.time()
    # Training
    with train_summary_writer.as_default():
        for idx, train_batch in enumerate(train_dataset):
            train_step(train_batch)
            if(idx % 500 == 0):
                board_logging(model, test_vector, step=optimizer.iterations)
    end_time = time.time()

    # Print stats
    print("Epoch: {}, time {}".format(
        epoch, end_time-start_time))

plt.show()
