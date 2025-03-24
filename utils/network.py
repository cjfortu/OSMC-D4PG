'''
## Network ##
# Defines the critic and actor networks for both D4PG and DDPG
'''

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.activations as tfka
import tensorflow.keras.initializers as tfki
import tensorflow.keras.layers as tfkl
import numpy as np
from utils.ops import dense, relu, tanh, batchnorm, softmax, add
from utils.l2_projection import _l2_project
from params import train_params


class Critic(tf.keras.Model):
    def __init__(self, state_dims, action_dims, dense1_size, dense2_size, final_layer_init,\
                 added_size, v_min, v_max, num_atoms, *args, **kwargs):
        # Used to calculate the fan_in of the state layer...
        #...(e.g. if state_dims is (3,2) fan_in should equal 6)
        self.state_dims = tf.experimental.numpy.prod(state_dims, dtype = tf.float32)
        self.action_dims = tf.experimental.numpy.prod(action_dims, dtype = tf.float32)
        self.dense1_size = tf.cast(dense1_size, dtype = tf.float32)
        self.dense2_size = tf.cast(dense2_size, dtype = tf.float32)
        self.final_layer_init = final_layer_init
        self.added_size = added_size
        self.v_min = tf.constant(v_min)
        self.v_max = tf.constant(v_max)
        self.num_atoms = tf.constant(num_atoms)

        super().__init__(*args, **kwargs)

        self.dense_st1 = tfkl.Dense(self.dense1_size, activation = tfka.relu,
            kernel_initializer =\
                tfki.RandomUniform(
                (-1/tf.math.sqrt(self.state_dims)),
                1/tf.math.sqrt(self.state_dims)),
            bias_initializer = \
                tfki.RandomUniform(
                (-1/tf.math.sqrt(self.state_dims)),
                1/tf.math.sqrt(self.state_dims)),
            )
        self.dense_st2 = tfkl.Dense(self.dense2_size, activation = tfka.relu,
            kernel_initializer =\
                tfki.RandomUniform(
                (-1/tf.math.sqrt(self.dense1_size + self.action_dims)),
                1/tf.math.sqrt(self.dense1_size + self.action_dims)),
            bias_initializer = \
                tfki.RandomUniform(
                (-1/tf.math.sqrt(self.dense1_size + self.action_dims)),
                1/tf.math.sqrt(self.dense1_size + self.action_dims)),
            )
        self.dense_act1 = tfkl.Dense(self.dense2_size, activation = tfka.relu,
            kernel_initializer =\
                tfki.RandomUniform(
                (-1/tf.math.sqrt(self.dense1_size + self.action_dims)),
                1/tf.math.sqrt(self.dense1_size + self.action_dims)),
            bias_initializer = \
                tfki.RandomUniform(
                (-1/tf.math.sqrt(self.dense1_size + self.action_dims)),
                1/tf.math.sqrt(self.dense1_size + self.action_dims)),
            )
        if train_params.MODE == 'OSMC-DDPG':
            self.dense_add1 = tfkl.Dense(self.added_size, activation = tfka.relu,
                kernel_initializer =\
                    tfki.RandomUniform(-1 * self.final_layer_init, self.final_layer_init),
                bias_initializer = \
                    tfki.RandomUniform(-1 * self.final_layer_init, self.final_layer_init),
                )
            self.dense_add2 = tfkl.Dense(1, activation = None,
                kernel_initializer =\
                    tfki.RandomUniform(-1 * self.final_layer_init, self.final_layer_init),
                bias_initializer = \
                    tfki.RandomUniform(-1 * self.final_layer_init, self.final_layer_init),
                )
        elif train_params.MODE == 'OSMC-D4PG':
            self.dense_add1 = tfkl.Dense(self.num_atoms, activation=None,
                kernel_initializer =\
                    tfki.RandomUniform(-1 * self.final_layer_init, self.final_layer_init),
                bias_initializer = \
                    tfki.RandomUniform(-1 * self.final_layer_init, self.final_layer_init),
                )


    def call(self, inputs):
        state = inputs[0]
        action = inputs[1]
        stateout = self.dense_st1(state)
        stateout = self.dense_st2(stateout)

        # Action path
        actionout = self.dense_act1(action)
        # Combined path
        output = tfkl.add([stateout, actionout])
        if train_params.MODE == 'OSMC-DDPG':
            output = self.dense_add1(output)
            output = self.dense_add2(output)
        elif train_params.MODE == 'OSMC-D4PG':
            output_logits = self.dense_add1(output)
            output_probs = tfka.softmax(output_logits)
            z_atoms = tf.linspace(self.v_min, self.v_max, self.num_atoms)
            Q_val = tf.reduce_sum(z_atoms * output_probs)

        if train_params.MODE == 'OSMC-DDPG':
            return output
        elif train_params.MODE == 'OSMC-D4PG':
            return output_logits, output_probs, z_atoms


class Actor(tf.keras.Model):
    def __init__(self, state_dims, action_dims, action_bound_low, action_bound_high,\
                 dense1_size, dense2_size, final_layer_init, *args, **kwargs):
        self.state_dims = tf.experimental.numpy.prod(state_dims, dtype = tf.float32)
        self.action_dims = tf.experimental.numpy.prod(action_dims, dtype = tf.float32)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.dense1_size = tf.cast(dense1_size, dtype = tf.float32)
        self.dense2_size = tf.cast(dense2_size, dtype = tf.float32)
        self.final_layer_init = final_layer_init

        super().__init__(*args, **kwargs)

        self.dense_1 = tfkl.Dense(self.dense1_size, activation=tfk.activations.relu,
            kernel_initializer =\
                tfki.RandomUniform(
                (-1/tf.math.sqrt(self.state_dims)),
                1/tf.math.sqrt(self.state_dims)),
            bias_initializer = \
                tfki.RandomUniform(
                (-1/tf.math.sqrt(self.state_dims)),
                1/tf.math.sqrt(self.state_dims)),
            )
        self.dense_2 = tfkl.Dense(self.dense2_size, activation=tfk.activations.relu,
            kernel_initializer =\
                tfki.RandomUniform(
                (-1/tf.math.sqrt(self.dense1_size)),
                1/tf.math.sqrt(self.dense1_size)),
            bias_initializer = \
                tfki.RandomUniform(
                (-1/tf.math.sqrt(self.dense1_size)),
                1/tf.math.sqrt(self.dense1_size)),
            )
        self.dense_3 = tfkl.Dense(self.action_dims, activation=tfk.activations.tanh,
            kernel_initializer =\
                tfki.RandomUniform(-1 * self.final_layer_init, self.final_layer_init),
            bias_initializer = \
                tfki.RandomUniform(-1 * self.final_layer_init, self.final_layer_init),
                )


    def call(self, inputs):
        state = inputs
        output = self.dense_1(state)
        output = self.dense_2(output)
        output = self.dense_3(output)

        output = tfkl.multiply([
                output,
                tf.broadcast_to(self.action_bound_high - self.action_bound_low, output.shape)])
        output = tfkl.add([
                output,
                tf.broadcast_to(self.action_bound_high + self.action_bound_low, output.shape)])
        output = tfkl.multiply([tf.broadcast_to(0.5, output.shape), output])

        return output
