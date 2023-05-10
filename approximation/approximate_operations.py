"""
8 bit approximation
"""
import tensorflow as tf
import numpy as np

""" CONV2D"""

class ApproxConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid'):
        super(ApproxConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        # Initialize weights and bias for the approximate Conv2D layer
        self.w_approx = self.add_weight(name='w_approx', shape=kernel_size+(input_shape[-1], filters), trainable=True)
        self.b_approx = self.add_weight(name='b_approx', shape=(filters,), trainable=True)

    def call(self, inputs):
        # Calculate the output of the approximate Conv2D layer using 8-bit approximate multipliers
        outputs = tf.nn.conv2d(inputs, self.w_approx, strides=self.strides, padding=self.padding)
        outputs = tf.nn.bias_add(outputs, self.b_approx)
        outputs = tf.quantization.fake_quant_with_min_max_args(outputs, min=-128, max=127, num_bits=8)

        return outputs


""" CONV3D"""

class ApproxConv3D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1, 1), padding='valid'):
        super(ApproxConv3D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        # Initialize weights and bias for the approximate Conv3D layer
        self.w_approx = self.add_weight(name='w_approx', shape=kernel_size+(filters,), trainable=True)
        self.b_approx = self.add_weight(name='b_approx', shape=(filters,), trainable=True)

    def call(self, inputs):
        # Calculate the output of the approximate Conv3D layer using 8-bit approximate multipliers
        outputs = tf.nn.conv3d(inputs, self.w_approx, strides=self.strides, padding=self.padding)
        outputs = tf.nn.bias_add(outputs, self.b_approx)
        outputs = tf.quantization.fake_quant_with_min_max_args(outputs, min=-128, max=127, num_bits=8)

        return outputs
        

""" Dense/Fully Connected Layer"""

# define the approximate multiplier function
def approx_mult(x, y, bitwidth):
    max_val = tf.constant(2**(bitwidth-1)-1, dtype=tf.float32)
    min_val = tf.constant(-2**(bitwidth-1), dtype=tf.float32)
    x = tf.clip_by_value(x, min_val, max_val)
    y = tf.clip_by_value(y, min_val, max_val)
    x = tf.cast(x, dtype=tf.int32)
    y = tf.cast(y, dtype=tf.int32)
    prod = tf.multiply(x, y)
    prod = tf.cast(prod, dtype=tf.float32)
    return prod

# define the dense layer with approximate multiplier
class DenseApprox(tf.keras.layers.Layer):
    def __init__(self, units, bitwidth=8, **kwargs):
        super(DenseApprox, self).__init__(**kwargs)
        self.units = units
        self.bitwidth = bitwidth

    def build(self, input_shape):
        self.w = self.add_weight(name='weights',
                                 shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        w = tf.clip_by_value(self.w, -1.0, 1.0)
        x = tf.cast(inputs, dtype=tf.float32)
        y = tf.cast(w, dtype=tf.float32)
        output = approx_mult(x, y, self.bitwidth)
        return output

# create a sample dense layer with approximate multiplier
dense_approx = DenseApprox(units=64, bitwidth=8)

# create a sample input
input_data = np.random.rand(32, 128)

# pass the input through the dense layer
output_data = dense_approx(input_data)

print("Input shape:", input_data.shape)
print("Output shape:", output_data.shape)



"""RELU Function"""

# Define the 8-bit approximate multiplier function
def approx_mult_8(x, w):
    # Convert x and w to 8-bit integers
    x_int = np.round(x * 255).astype(np.uint8)
    w_int = np.round(w * 255).astype(np.uint8)
    # Perform multiplication using uint16 to avoid overflow
    mult_int = np.uint16(x_int) * np.uint16(w_int)
    # Convert back to float and divide by 2^16
    mult_float = mult_int.astype(np.float32) / 65536
    # Round to nearest integer
    mult_rounded = np.round(mult_float)
    # Convert back to 8-bit and return as float
    return mult_rounded.astype(np.uint8) / 255

# Define the ReLU function using the approximate multiplier
def relu_approx(x, w):
    return np.maximum(0, approx_mult_8(x, w))

# Example usage with random input and weight
x = np.random.randn(10, 10).astype(np.float32)
w = np.random.randn(10, 10).astype(np.float32)
y = relu_approx(x, w)

