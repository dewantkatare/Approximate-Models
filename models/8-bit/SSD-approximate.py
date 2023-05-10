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

import tensorflow as tf

class ApproxConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ApproxConv2D, self).__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        super(ApproxConv2D, self).build(input_shape)

        self.alpha = tf.Variable(tf.ones(shape=(1,), dtype=tf.float32), trainable=True, name='alpha')
        self.scale = tf.Variable(tf.ones(shape=(1,), dtype=tf.float32), trainable=True, name='scale')

    def call(self, inputs):
        quantized_weights = tf.quantization.quantize(self.kernel, self.scale, self.alpha, mode='SCALED')
        inputs = tf.quantization.quantize(inputs, 1.0, 0, mode='SCALED')

        outputs = tf.nn.conv2d(inputs, quantized_weights, strides=self.strides, padding=self.padding.upper(), data_format='NHWC')
        outputs = tf.quantization.dequantize(outputs, self.scale, self.alpha)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
            outputs = tf.quantization.dequantize(outputs, self.scale, self.alpha)

        return self.activation(outputs)

# define SSD model with 8-bit approximate multiplier
def create_ssd_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = ApproxConv2D(64, 3, activation='relu', padding='same')(inputs)
    x = ApproxConv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = ApproxConv2D(128, 3, activation='relu', padding='same')(x)
    x = ApproxConv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = ApproxConv2D(256, 3, activation='relu', padding='same')(x)
    x = ApproxConv2D(256, 3, activation='relu', padding='same')(x)
    x = ApproxConv2D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = ApproxConv2D(512, 3, activation='relu', padding='same')(x)
    x = ApproxConv2D(512, 3, activation='relu', padding='same')(x)
    x = ApproxConv2D(512, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = ApproxConv2D(512, 3, activation='relu', padding='same')(x)
    x = ApproxConv2D(512, 3, activation='relu', padding='same')(x)
    x = ApproxConv2D(512, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
