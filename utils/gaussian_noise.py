'''
## Gaussian Noise ##
# Creates Gaussian noise process for adding exploration noise to the action space during training
@author: Mark Sinton (msinto93@gmail.com) .
'''

import tensorflow as tf

class GaussianNoiseGenerator:
    def __init__(self, action_dims, action_bound_low, action_bound_high, noise_scale):
        # assert np.array_equal(np.abs(action_bound_low), action_bound_high)

        self.action_dims = action_dims
        self.action_bounds = (action_bound_high - action_bound_low) / 2
        self.scale = noise_scale

    def __call__(self):
        noise = tf.random.normal(shape = self.action_dims) * self.action_bounds * self.scale

        return noise


