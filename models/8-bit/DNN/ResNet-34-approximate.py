#from resnet34 import ResNet34

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from resnet34_approx import ResNet34_approx, approx_multiply

# define 8-bit approximate multiplier
@tf.custom_gradient
def approx_multiply(x, y):
    scale_factor = tf.constant(0.5, dtype=tf.float32)
    out = tf.multiply(x, y)
    out = tf.round(tf.multiply(out, scale_factor))
    
    def grad(dy):
        grad_x = tf.multiply(dy, y)
        grad_y = tf.multiply(dy, x)
        return grad_x, grad_y
        
    return out, grad

# define ResNet-34 model with 8-bit approximate multiplier
def ResNet34_approx(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    x = ResNetBlock(x, filters=[64,64], strides=(1,1))
    x = ResNetBlock(x, filters=[64,64], strides=(1,1))
    x = ResNetBlock(x, filters=[128,128], strides=(2,2))
    x = ResNetBlock(x, filters=[128,128], strides=(1,1))
    x = ResNetBlock(x, filters=[256,256], strides=(2,2))
    x = ResNetBlock(x, filters=[256,256], strides=(1,1))
    x = ResNetBlock(x, filters=[512,512], strides=(2,2))
    x = ResNetBlock(x, filters=[512,512], strides=(1,1))
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=num_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model

# define ResNet block with 8-bit approximate multiplier
def ResNetBlock(x, filters, strides):
    shortcut = x
    x = Conv2D(filters=filters[0], kernel_size=(3,3), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters[1], kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x, max_value=6.0)  # apply ReLU6 activation
    return x


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
model = ResNet34_approx(input_shape=input_shape, num_classes=num_classes)

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
model.save('resnet34_approx_imagenet.h5')


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