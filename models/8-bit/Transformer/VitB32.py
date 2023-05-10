import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from vit_models import vit_b32

# Load the nuScenes dataset
ds, info = tfds.load('nuscenes', split='train', with_info=True)

# Preprocess the data
def preprocess(features):
    # Resize and normalize image
    image = tf.image.resize(features['image'], (224, 224)) / 255.
    # Convert target to one-hot encoding
    target = tf.one_hot(features['label']['category'], depth=10)
    # Normalize bounding box coordinates
    bbox = features['label']['box'] / 700.
    # Combine target and bbox into single tensor
    label = tf.concat([target, bbox], axis=-1)
    return image, label

ds = ds.map(preprocess).batch(32)

# Define the model
def create_model():
    model = vit_b32(
        image_size=224,
        activation='softmax',
        classes=10,
        include_top=True,
        weights=None
    )
    return model

model = create_model()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    loss={'output_1': 'categorical_crossentropy', 'output_2': 'mse'},
    metrics={'output_1': 'accuracy', 'output_2': 'mae'}
)

# Train the model
model.fit(ds, epochs=10)
