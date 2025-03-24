'''
## Test ##
# Test a trained D4PG network. This can be run alongside training by running 'test_every_new_ckpt.py'.
@author: Mark Sinton (msinto93@gmail.com)
'''

import tensorflow as tf
import numpy as np
import time


from params import test_params
from agent import Agent


def test():
    start_time = time.time()
    # Set random seeds for reproducability
    np.random.seed(test_params.RANDOM_SEED)
    # tf.compat.v1.set_random_seed(test_params.RANDOM_SEED)
    tf.random.set_seed(test_params.RANDOM_SEED)

    # # Create session
    # config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=config)

    # Initialise agent
    agent = Agent(test_params.ENV, test_params.RANDOM_SEED, None, start_time)
    # Build network
    agent.build_network(training=False)

    # Test network
    agent.test()

    # sess.close()


if  __name__ == '__main__':
    test()






