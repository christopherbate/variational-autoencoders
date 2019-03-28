# Convolutional Variational Autoencoder, Tensorflow 2.0a Implmentation

This repository contains an implementation of variational autoencoders with convolutional encoder/decoder,
where everything except the convolutional part is based on the paper [1] below.

The implementation is based on the example in Tensorflow documentation [2]. 

Some key differences/additions: 
1. Cleaned up code to make it more readable. 
2. Added tf.function highlights for speed performance
3. Replaced the KL term calculations with a term which is directly calculated, as shown 
in the paper. This speeds up the training and is actually simpler than what the code originally had. 
4. Added weight checkpoints, metrics, Tensorboard functionality that allows you to visualize 
conv layers, generative samples, and so on.

# Setup

The following script was used to create a Tensorflow 2.0-alpha gpu environment. Some of the instructions 
on the website appear incorrect regarding which packages to install. Note you need CUDA 10.0, and CUDA 10.1 
will not work with TF package (2.0 or otherwise).

```
virtualenv -p python3 ./tf-2-gpu-env
pip install tensorflow-gpu==2.0.0a
pip install matplotlib
```

# Training 

Make sure you are in Python3 with TF2.0 activated.

1. `python vae_mnist.py`
2. `tensorboard --logdir=./logs/`

# References
[1]D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” arXiv:1312.6114 [cs, stat], Dec. 2013.
[2]https://www.tensorflow.org/alpha/tutorials/generative/cvae