# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:51:25 2023

@author: dewantkatare
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16

def ssd_vgg16(num_classes):
    # Load VGG16 as the base network
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Add additional convolutional layers on top of the base model
    x = base_model.output
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv6_1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv6_2')(x)
    x = layers.Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv7_1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv7_2')(x)

    # Add additional convolutional layers for object detection
    num_anchors = 4

    # Replace the original 3x3 convolution with an 8-bit approximation using the XNOR-Net method
    kernel_bits = 8
    weights = base_model.layers[-1].kernel
    xnor_weights = tf.round(tf.clip_by_value(weights, -1, 1) * ((2**(kernel_bits-1))-1))
    xnor_weights = tf.cast(tf.sign(xnor_weights), dtype=tf.float32)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=(1, 1), padding='same', depth_multiplier=num_anchors*(num_classes+4),
                               use_bias=False, weights=[xnor_weights])(x)
    x = layers.Activation('relu')(x)

    # Reshape the output to match the expected shape of the model's final output
    x = layers.Reshape((-1, num_classes + 4))(x)

    # Create the SSD model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

    return model




# Instantiate the model
num_classes = 4 # number of object classes
input_shape = (224, 224, 3) # input image shape
model = ssd_vgg16(num_classes)

# Compile the model
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
losses = {
    'box_output': 'mse',
    'class_output': 'categorical_crossentropy'
}
loss_weights = {
    'box_output': 1.0,
    'class_output': 1.0
}
metrics = {
    'box_output': 'mse',
    'class_output': 'accuracy'
}
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

# Print the model summary
model.summary()