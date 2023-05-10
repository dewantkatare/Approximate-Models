# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:51:25 2023

@author: dewantkatare
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Flatten, Dense

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, ImageDataGenerator

import numpy as np

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = preprocess_input(x_train.astype('float32'))
x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, axis=-1))
y_train = to_categorical(y_train, num_classes=10)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

# Reshape images
x_train = tf.image.resize(x_train, (224, 224))


def conv_block(x, filters, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding=padding)(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding)(x)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def resnet18(input_shape=(224, 224, 3), num_classes=1000):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, filters=64, kernel_size=(3, 3))
    x = conv_block(x, filters=64, kernel_size=(3, 3))
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = conv_block(x, filters=128, kernel_size=(3, 3))
    x = conv_block(x, filters=128, kernel_size=(3, 3))
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = conv_block(x, filters=256, kernel_size=(3, 3))
    x = conv_block(x, filters=256, kernel_size=(3, 3))
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = conv_block(x, filters=512, kernel_size=(3, 3))
    x = conv_block(x, filters=512, kernel_size=(3, 3))
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, x, name='resnet18')

    return model

model = resnet18()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)