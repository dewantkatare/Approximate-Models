# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:51:25 2023

@author: dewantkatare
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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



def approx_conv2d(inputs, filters, kernel_size, strides, padding):
    scale_factor = 127.0 / np.max(np.abs(inputs))
    inputs = inputs * scale_factor
    filters = filters * scale_factor
    output = tf.nn.conv2d(inputs, filters, strides, padding)
    output = output / scale_factor
    return output

def approx_dense(inputs, weights):
    scale_factor = 127.0 / np.max(np.abs(inputs))
    inputs = inputs * scale_factor
    weights = weights * scale_factor
    output = tf.matmul(inputs, weights)
    output = output / scale_factor
    return output

def create_approx_vgg16():
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Classification layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))

    # Convert all the conv and dense layers to 8-bit approximate layers
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            layer.kernel = tf.Variable(approx_conv2d(layer.input, layer.kernel, layer.kernel_size, layer.strides, layer.padding))
        elif isinstance(layer, Dense):
            layer.kernel = tf.Variable(approx_dense(layer.input, layer.kernel))
    return model

# Create the model and compile it
model = create_approx_vgg16()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'], experimental_run_tf_function=False)

# Print the model summary
model.summary()

# Train the model
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10, steps_per_epoch=len(x_train) // 32)
