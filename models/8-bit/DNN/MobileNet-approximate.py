# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:51:25 2023

@author: dewantkatare
"""

import tensorflow as tf

class ApproximateMultiplier(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApproximateMultiplier, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,),
                                      initializer=tf.keras.initializers.Constant(value=0.5),
                                      trainable=False)
        super(ApproximateMultiplier, self).build(input_shape)

    def call(self, inputs):
        approx_inputs = tf.cast(inputs * 255.0, tf.uint8)
        approx_inputs = tf.cast(approx_inputs, tf.float32) / 255.0
        approx_outputs = tf.multiply(approx_inputs, approx_inputs)
        approx_outputs = tf.cast(approx_outputs * 255.0, tf.uint8)
        approx_outputs = tf.cast(approx_outputs, tf.float32) / 255.0
        outputs = tf.multiply(approx_outputs, self.kernel)
        return outputs

# Define the input shape and number of classes
input_shape = (224, 224, 3)
num_classes = 1000

model = tf.keras.applications.MobileNet(input_shape=input_shape, include_top=True, weights=None, classes=num_classes)

# Add the approximate multiplier layer
model.add(ApproximateMultiplier())

# Define the training parameters
batch_size = 32
epochs = 10
learning_rate = 1e-4

# Define the path to the ImageNet dataset
data_dir = '/path/to/imagenet/'

# Load the ImageNet training data using the tf.keras.preprocessing module
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=validation_generator)