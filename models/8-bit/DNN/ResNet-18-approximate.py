# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:51:25 2023

@author: dewantkatare
"""

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Add, Flatten, Dense,)
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = tf.image.resize(x_train, size=(32, 32)).numpy() # Resize images to match input shape of ResNet18
x_train = x_train.astype('float32') / 255.0 # Normalize pixel values to [0,1]
y_train = to_categorical(y_train)

def conv2d_8bit(filters, kernel_size, strides=(1, 1), padding='same'):
    return Conv2D( filters, kernel_size, strides=strides, padding=padding, activation=None, use_bias=False, dtype=tf.uint8,)

def dense_8bit(units):
    return Dense(
        units,
        activation=None,
        use_bias=False,
        dtype=tf.uint8,
    )

def resnet_block(x, filters, kernel_size, strides=(1, 1), padding='same', activation=True):
    shortcut = x

    x = conv2d_8bit(filters, kernel_size, strides, padding)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)

    x = conv2d_8bit(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)

    if strides != (1, 1) or shortcut.shape[-1] != filters:
        shortcut = conv2d_8bit(filters, (1, 1), strides, padding='valid')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)

    return x

def ResNet18_approx(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = conv2d_8bit(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = resnet_block(x, 64, (3, 3))
    x = resnet_block(x, 64, (3, 3))
    x = resnet_block(x, 128, (3, 3), strides=(2, 2))
    x = resnet_block(x, 128, (3, 3))
    x = resnet_block(x, 256, (3, 3), strides=(2, 2))
    x = resnet_block(x, 256, (3, 3))
    x = resnet_block(x, 512, (3, 3), strides=(2, 2))
    x = resnet_block(x, 512, (3, 3))

    x = Flatten()(x)
    x = dense_8bit(512)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = dense_8bit(num_classes)(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.softmax(x)

    model = Model(inputs=inputs, outputs=x)
    return model

#Build model
model = ResNet18_approx(input_shape=(32, 32, 1), num_classes=10)
# Print the model summary
model.summary()

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

