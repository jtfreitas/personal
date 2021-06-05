"""
Training GAN to learn and reproduce handwritten digits, based on MNIST.

The script returns a hd5 with model parameters after achieving an accuracy of 99%

- João Freitas
"""
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam

import matplotlib.pyplot as plt

from tensorflow import random, ones_like, zeros_like

def generator_model():
    
    model = Sequential()
    model.add(Dense(7*7*256))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Reshape((7, 7, 256)))
   # assert model.output_shape == (None, 7, 7, 256)
    
    model.add(Conv2DTranspose(128, (5, 5),
                              strides = (1, 1),
                              padding = 'same',
                              use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(64, (5, 5),
                                     strides = (2, 2),
                                     padding = 'same',
                                     use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(64, (5, 5),
                              strides = (2, 2),
                              padding = 'same',
                              use_bias = False))
    return model

gen = generator_model()

def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5,5),
                     strides = (2,2),
                     padding = 'same',
                     input_shape = [28,28,1]))
    
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, (5,5),
                     strides = (2,2),
                     padding = 'same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(1))
    
    return model

disc = discriminator_model()

cross_entropy = BinaryCrossentropy(from_logits = True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(ones_like(real_output), real_output)
    fake_loss = cross_entropy(zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(ones_like(fake_output), fake_output)

gen_optimizer, disc_optimizer = Adam(1e-4), Adam(1e-4)

# Load, reshape, normalize dataset.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32'), x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
x_train, x_test = (x_train - 127.5)/127.5 , (x_test - 127.5)/127.5

epochs = 50

noise_dim = 100
gen_samples = 16

seed = random.normal([gen_samples, noise_dim])