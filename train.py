'''
## Train ##
# Code to train OSMC-D4PG and OSMC-DDPG on custom tank battle environment
@author: Clemente Fortuna (clemente.fortuna@mtsi-va.com)

multithreaded approach from: https://github.com/msinto93/D4PG
distributional RL from: https://arxiv.org/pdf/1707.06887
one step two critic from: https://arxiv.org/pdf/2203.16289
'''
import threading
import random
import tensorflow as tf
import numpy as np
import time

from params import train_params
from utils.prioritised_experience_replay import PrioritizedReplayBuffer
# from utils.gaussian_noise import GaussianNoiseGenerator
from agent import Agent
from learner import Learner

from tensorflow.python.framework.ops import disable_eager_execution


def train():
    tf.keras.backend.clear_session()
    start_time = time.time()

    # disable_eager_execution()
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.config.optimizer.set_jit(True)

    # Set random seeds for reproducability
    np.random.seed(train_params.RANDOM_SEED)
    random.seed(train_params.RANDOM_SEED)
    tf.random.set_seed(train_params.RANDOM_SEED)

    # Initialise prioritised experience replay memory
    PER_memory = PrioritizedReplayBuffer(
            train_params.REPLAY_MEM_SIZE, train_params.PRIORITY_ALPHA)

    # # Initialise Gaussian noise generator
    # gaussian_noise = GaussianNoiseGenerator(
    #         train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW,\
    #         train_params.ACTION_BOUND_HIGH, train_params.NOISE_SCALE)

    # Create threads for learner process and agent processes
    threads = []

    # Create threading events for communication and synchronisation...
    #...between the learner and agent threads
    run_agent_event = threading.Event()
    stop_agent_event = threading.Event()

    # with tf.device('/device:GPU:0'):
    # Initialise learner
    learner = Learner(PER_memory, run_agent_event, stop_agent_event)

    # Build learner networks
    learner.build_networks()

    # Initialise variables (either from ckpt file if given, or from random)
    learner.initialise_vars()

    threads.append(threading.Thread(target=learner.run))


    for n_agent in range(train_params.NUM_AGENTS):
        # Initialise agent)
        agent = Agent(train_params.ENV, train_params.RANDOM_SEED, learner, start_time,\
                      PER_memory, run_agent_event, stop_agent_event, n_agent)
        # Build network
        agent.build_network(training=True)

        # Create Tensorboard summaries to save episode rewards
        if train_params.LOG_DIR is not None:
            agent.build_summaries(train_params.LOG_DIR + ('/agent_%02d' % n_agent))

        threads.append(threading.Thread(target=agent.run))

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if  __name__ == '__main__':
    # print(tf.test.is_gpu_available(cuda_only=True))
    train()





