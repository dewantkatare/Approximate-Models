import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
#from resnet34 import ResNet34


import tensorflow as tf

def conv_block(x, filters, kernel_size, strides, padding='same', use_bias=False):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def identity_block(x, filters):
    shortcut = x
    x = conv_block(x, filters=filters, kernel_size=3, strides=1)
    x = conv_block(x, filters=filters, kernel_size=3, strides=1)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

def projection_block(x, filters, strides):
    shortcut = x
    x = conv_block(x, filters=filters, kernel_size=3, strides=strides)
    x = conv_block(x, filters=filters, kernel_size=3, strides=1)
    shortcut = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=strides, padding='same', use_bias=False)(shortcut)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

def ResNet34(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = conv_block(inputs, filters=64, kernel_size=7, strides=2)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

    x = projection_block(x, filters=64, strides=1)
    x = identity_block(x, filters=64)
    x = identity_block(x, filters=64)

    x = projection_block(x, filters=128, strides=2)
    x = identity_block(x, filters=128)
    x = identity_block(x, filters=128)
    x = identity_block(x, filters=128)

    x = projection_block(x, filters=256, strides=2)
    x = identity_block(x, filters=256)
    x = identity_block(x, filters=256)
    x = identity_block(x, filters=256)
    x = identity_block(x, filters=256)
    x = identity_block(x, filters=256)

    x = projection_block(x, filters=512, strides=2)
    x = identity_block(x, filters=512)
    x = identity_block(x, filters=512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

input_shape = (224, 224, 3)
num_classes = 1000

#model = ResNet34(input_shape, num_classes)

#model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
              
# set batch size, number of classes, and number of epochs
batch_size = 32

epochs = 50



# create a ResNet-34 model
model = ResNet34(input_shape=input_shape, num_classes=num_classes)

# set optimizer and loss function
optimizer = SGD(lr=0.1, momentum=0.9, decay=1e-4)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# set up data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# set up data generator for training and validation
train_generator = train_datagen.flow_from_directory(
    'path/to/ImageNet/train/',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'path/to/ImageNet/val/',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# define learning rate schedule
def lr_schedule(epoch):
    lr = 0.1
    if epoch > 30:
        lr /= 10
    elif epoch > 20:
        lr /= 5
    elif epoch > 10:
        lr /= 2
    print('Learning rate: ', lr)
    return lr

# define callbacks
lr_scheduler = LearningRateScheduler(lr_schedule)
checkpoint = ModelCheckpoint('resnet34_imagenet.h5', monitor='val_accuracy', save_best_only=True)

# train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.n // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.n // batch_size,
    callbacks=[lr_scheduler, checkpoint]
)

# save the model
model.save('resnet34_imagenet.h5')


"""


- /ImageNet/
    - train/
        - class1/
            - image1.jpg
            - image2.jpg
            ...
        - class2/
            - image1.jpg
            - image2.jpg
            ...
        ...
    - val/
        - class1/
            - image1.jpg
            - image2.jpg
            ...
        - class2/
            - image1.jpg
            - image2.jpg
            ...
        ...
"""