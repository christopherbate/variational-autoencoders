'''
Conv VAE model
'''
import tensorflow as tf


class CVAE(tf.keras.Model):
    '''
    Conv VAE
    Model assumptions are:
    1) Prior is gaussian
    2) Approximate posterior q(z|x_sample) is gaussian
    3) Vectors mu, sigma are the outputs of q(z|x_sample) - the "inference_net"
    4) 
    '''

    def __init__(self, latent_dim, img_shape=(28, 28, 1)):
        super(CVAE, self).__init__(name='cvae')
        self.latent_dim = latent_dim

        '''
        Approximate posterior network
        '''
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=img_shape),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2),
                    activation='relu'
                ),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2),
                    activation='relu'
                ),
                tf.keras.layers.Flatten(),
                # Last layer - first half are the means, second half are
                # the log variances
                tf.keras.layers.Dense(latent_dim+latent_dim)
            ]
        )

        '''
        Decoder network aka generative network
        '''
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation='relu'),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME")
            ]
        )

    def encode(self, x):
        # split along axis 1. axis 0 is batch size
        inference = self.inference_net(x)
        mean, logvar = tf.split(self.inference_net(x), 2, axis=1)
        return mean, logvar

    def reparamaterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps*tf.exp(logvar*0.5)+mean

    def decode(self, z):
        estimate = self.generative_net(z)
        # We return the logits estimate, tf will apply sigmoid in loss
        return estimate


def compute_loss(model, x):
    '''
    Computes the ELBO loss given an example x
    Assumptions, 
    1. prior on z is
    sphereical gaussian mean 0 covariance I
    2. p(x|z) is bernoili, where we are predicting
    pixels each individually as being zero or 1
    3. log q(z|x) = log N(u(i),sigma(i))
    '''
    mean, logvar = model.encode(x)
    z = model.reparamaterize(mean, logvar)

    # print("Mean: ", mean[0])
    # print("Logvar: ", logvar[0])
    # print("Z: ", z[0])

    # print("Reparam shape: ", z.shape)

    logits = model.decode(z)

    # print("Logits shape: ", logits.shape)

    # This is a component-wise logistic loss
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=x)

    # Sum the losses
    reconstruction_loss = tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    # print("Reconstruction Loss shape: ", reconstruction_loss.shape)

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
    regularization_loss = -0.5*(1+logvar-tf.square(mean)-tf.exp(logvar))
    regularization_loss = tf.reduce_sum(regularization_loss, axis=1)

    # print("Regularization Loss shape: ", regularization_loss.shape)
    loss = tf.reduce_mean(regularization_loss + reconstruction_loss)
    # print(loss.shape)
    return loss
