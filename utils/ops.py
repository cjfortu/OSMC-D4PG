'''
## Ops ##
# Common ops for the networks
@author: Mark Sinton (msinto93@gmail.com)
'''

import tensorflow as tf

class add(tf.keras.layers.Layer):
    def __init__(self, scope='add',\
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scope = scope

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.scope):
            out = tf.keras.layers.add(inputs)
        return out

# def conv2d(inputs, kernel_size, filters, stride, activation=None, use_bias=True, weight_init=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), bias_init=tf.compat.v1.zeros_initializer(), scope='conv'):
#     with tf.compat.v1.variable_scope(scope):
#         if use_bias:
#             return tf.compat.v1.layers.conv2d(inputs, filters, kernel_size, stride, 'valid', activation=activation,
#                                     use_bias=use_bias, kernel_initializer=weight_init)
#         else:
#             return tf.compat.v1.layers.conv2d(inputs, filters, kernel_size, stride, 'valid', activation=activation,
#                                     use_bias=use_bias, kernel_initializer=weight_init,
#                                     bias_initializer=bias_init)


class conv2d(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters, stride, activation=None, use_bias=True,\
            weight_init=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform"),\
            bias_init=tf.compat.v1.zeros_initializer(), scope='conv',\
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.strid = stride
        self.activation = activation
        self.use_bias = use_bias
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.scope = scope

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.scope):
            if use_bias:
                return tf.compat.v1.layers.conv2d(inputs, self.filters, self.kernel_size, self.stride,\
                        'valid', activation=self.activation, use_bias=self.use_bias,\
                        kernel_initializer=self.weight_init)
            else:
                return tf.compat.v1.layers.conv2d(inputs, self.filters, self.kernel_size, self.stride,\
                        'valid', activation=self.activation, use_bias=self.use_bias,\
                        kernel_initializer=self.weight_init, bias_initializer=self.bias_init)


# def batchnorm(inputs, is_training, momentum=0.9, scope='batch_norm'):
#     with tf.compat.v1.variable_scope(scope):
#         return tf.compat.v1.layers.batch_normalization(inputs, momentum=momentum, training=is_training, fused=True)


class batchnorm(tf.keras.layers.Layer):
    def __init__(self, is_training, momentum=0.9, scope='batch_norm', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_training = training
        self.momentum = momentum
        self.scope = scope

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.scope):
            return tf.compat.v1.layers.batch_normalization(inputs, momentum=self.momentum,\
                    training=self.is_training, fused=True)


# def dense(inputs, output_size, activation=None, weight_init=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), bias_init=tf.compat.v1.zeros_initializer(), scope='dense'):
#     with tf.compat.v1.variable_scope(scope):
#         return tf.compat.v1.layers.dense(inputs, output_size, activation=activation, kernel_initializer=weight_init, bias_initializer=bias_init)


class dense(tf.keras.layers.Layer):
    def __init__(self, output_size, activation=None,\
            weight_init=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"),\
            bias_init=tf.compat.v1.zeros_initializer(), scope='dense',\
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.scope = scope

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.scope):
            out = tf.compat.v1.layers.dense(inputs, self.output_size, activation=self.activation,\
                    kernel_initializer=self.weight_init, bias_initializer=self.bias_init)
        return out


# def flatten(inputs, scope='flatten'):
#     with tf.compat.v1.variable_scope(scope):
#         return tf.compat.v1.layers.flatten(inputs)


class flatten(tf.keras.layers.Layer):
    def __init__(self, scope='flatten', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scope = scope

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.scope):
            return tf.compat.v1.layers.flatten(inputs)


# def relu(inputs, scope='relu'):
#     with tf.compat.v1.variable_scope(scope):
#         return tf.nn.relu(inputs)


class relu(tf.keras.layers.Layer):
    def __init__(self, scope='relu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scope = scope

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.scope):
            return tf.compat.v1.nn.relu(inputs)


# def tanh(inputs, scope='tanh'):
#     with tf.compat.v1.variable_scope(scope):
#         return tf.nn.tanh(inputs)


class tanh(tf.keras.layers.Layer):
    def __init__(self, scope='relu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scope = scope

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.scope):
            return tf.compat.v1.math.tanh(inputs)


# def softmax(inputs, scope='softmax'):
#     with tf.compat.v1.variable_scope(scope):
#         return tf.nn.softmax(inputs)


class softmax(tf.keras.layers.Layer):
    def __init__(self, scope='softmax', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scope = scope

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.scope):
            return tf.compat.v1.math.softmax(inputs)
