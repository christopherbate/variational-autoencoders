'''
Conv VAE model
'''
import tensorflow as tf
from tensorflow.keras import layers


class Sampling(tf.keras.layers.Layer):
    '''
    Used by the inference network to produce a value for the latent vector
    '''

    def call(self, inputs):
        latent_mean = inputs[0]
        latent_log_var = inputs[1]
        batch = tf.shape(latent_mean)[0]
        dim = tf.shape(latent_mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return latent_mean + tf.exp(0.5*latent_log_var)*eps


class InferenceNet(tf.keras.layers.Layer):
    ''' 
    Inference layer/sub-network
    The output is a vector of means and log variances, both of size latent_dim
    Also output the "sampled" latent variable computed from mean and log var
    '''

    def __init__(self, latent_dim, name='InferenceNet', **kwargs):
        super(InferenceNet, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.latent_means_layer = tf.keras.layers.Dense(latent_dim)
        self.latent_log_var_layer = tf.keras.layers.Dense(latent_dim)
        self.samplingLayer = Sampling()

    def call(self, inputs):
        result = self.conv1(inputs)
        result = self.conv2(result)
        flattened = self.flatten(result)
        latent_means = self.latent_means_layer(flattened)
        latent_log_var = self.latent_log_var_layer(flattened)
        sampled = self.samplingLayer((latent_means, latent_log_var))

        '''
           This is the first term of the loss, the KL term KL( q(z|x) || p(z) )
           I refer to this as the regularization loss
           because intuitively this term
           is penalizing the variational approximation of z
           from being far from the prior.
           In this case, the prior is simply that
           the mean is zero and the variance is 1,
           uncorrelated among the components.
           '''
        regularization_loss = -0.5 * tf.reduce_mean(
            latent_log_var - tf.square(latent_means) - tf.exp(latent_log_var)+1)
        self.add_loss(regularization_loss)
        return sampled


class GenerativeNet(tf.keras.layers.Layer):
    ''' 
    Generative layer/sub-network
    The output is a set of logits.
    '''

    def __init__(self, latent_dim, name='GenerativeNet', **kwargs):
        super(GenerativeNet, self).__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(units=7*7*32, activation='relu')
        self.reshape_layer = tf.keras.layers.Reshape(target_shape=(7, 7, 32))
        self.conv1_t = tf.keras.layers.Conv2DTranspose(filters=64,
                                                       kernel_size=3,
                                                       strides=(2, 2),
                                                       activation='relu',
                                                       padding='SAME')
        self.conv2_t = tf.keras.layers.Conv2DTranspose(filters=32,
                                                       kernel_size=3,
                                                       strides=(2, 2),
                                                       activation='relu',
                                                       padding='SAME')
        self.conv3_t = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(1, 1), padding='SAME',
        )

    def call(self, inputs):
        result = self.dense(inputs)
        result = self.reshape_layer(result)
        result = self.conv1_t(result)
        result = self.conv2_t(result)
        result = self.conv3_t(result)
        return result


class CVAE(tf.keras.Model):
    '''
    Conv VAE
    Model assumptions are:
    1) Prior is gaussian
    2) Approximate posterior q(z|x_sample) is gaussian
    3) Vectors mu, sigma are the outputs of q(z|x_sample) - the "inference_net"
    '''

    def __init__(self, latent_dim, name="CVAE", **kwargs):
        super(CVAE, self).__init__(name=name, **kwargs)
        self.infNet = InferenceNet(latent_dim=latent_dim)
        self.genNet = GenerativeNet(latent_dim=latent_dim)

    def call(self, inputs):
        # Inference
        latent_sample = self.infNet(inputs)

        # Re-generation
        generated = self.genNet(latent_sample)

        return generated


def vae_loss(y_true, y_pred):
