import tensorflow as tf
from tensorflow.keras import layers

# Define the SSD model architecture
def SSD(input_shape, num_classes):
    # Define the input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Define the VGG16 base network
    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    conv4 = base_model.get_layer('block4_conv3').output
    conv7 = base_model.get_layer('block5_conv3').output

    # Define the extra feature layers
    conv8_1 = layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7)
    conv8_2 = layers.Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv8_2')(conv8_1)

    conv9_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8_2)
    conv9_2 = layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv9_2')(conv9_1)

    conv10_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='conv10_1')(conv9_2)
    conv10_2 = layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv10_2')(conv10_1)

    conv11_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', name='conv11_1')(conv10_2)
    conv11_2 = layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv11_2')(conv11_1)

    # Define the output layers
    box_output = layers.Conv2D(num_classes * 4, (3, 3), padding='same', name='box_output')(conv7)
    class_output = layers.Conv2D(num_classes, (3, 3), padding='same', name='class_output')(conv7)

    # Define the SSD model
    model = tf.keras.Model(inputs=base_model.input, outputs=[box_output, class_output])
    return model

# Instantiate the model
num_classes = 4 # number of object classes
input_shape = (224, 224, 3) # input image shape
model = SSD(input_shape, num_classes)

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