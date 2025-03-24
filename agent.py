'''
## Agent ##
Agent class - the agent explores the environment, collecting experiences and adding them...
...to the PER buffer. Can also be used to test/run a trained network in the environment.
@author: Clemente Fortuna (clemente.fortuna@mtsi-va.com)

based on https://github.com/msinto93/D4PG
'''

import os
import sys
import tensorflow as tf
import numpy as np
import sonnet as snt
import scipy.stats as ss
from collections import deque
import cv2
import imageio
import pprint
import time

from params import train_params, test_params, play_params
from utils.network import Actor
from utils.env_wrapper import PendulumWrapper, LunarLanderContinuousWrapper,\
     BipedalWalkerWrapper
from utils.tank_env import Tank_Env


class Agent:

    def __init__(self, env, seed, learner, start_time, PER_memory,\
                 run_agent_event, stop_agent_event, n_agent=0):
        print("Initialising agent %02d... \n" % n_agent)

        self.learner = learner
        self.n_agent = n_agent
        self.start_time = start_time

        # Create environment
        if env == 'Pendulum-v0':
            self.env_wrapper = PendulumWrapper()
        elif env == 'LunarLanderContinuous-v2':
            self.env_wrapper = LunarLanderContinuousWrapper()
        elif env == 'BipedalWalker-v2':
            self.env_wrapper = BipedalWalkerWrapper()
        elif env == 'BipedalWalkerHardcore-v2':
            self.env_wrapper = BipedalWalkerWrapper(hardcore=True)
        elif env == 'Tank_Env_2D-2crit':
            self.env_wrapper = Tank_Env(mode = '2D-2crit', seed = seed*(n_agent+1))
        else:
            raise Exception('Chosen environment does not have an environment wrapper' +\
                            'defined. Please choose an environment with an environment' +\
                            'wrapper defined, or create a wrapper for this environment in' +\
                            'utils.env_wrapper.py')
        # self.env_wrapper.set_random_seed(seed*(n_agent+1))

        self.PER_memory = PER_memory
        self.run_agent_event = run_agent_event
        self.stop_agent_event = stop_agent_event
        self.n_agent = n_agent

        self.noise_decay = tf.constant(train_params.NOISE_DECAY, dtype = tf.float32)
        self.action_bound_low = tf.constant(train_params.ACTION_BOUND_LOW, dtype = tf.float32)
        self.action_bound_high = tf.constant(train_params.ACTION_BOUND_HIGH, dtype = tf.float32)
        self.noise_std = tf.constant(train_params.NOISE_STD, dtype = tf.float32)


    def build_network(self, training):

        if training:
            # each agent has their own var_scope
            var_scope = ('actor_agent_%02d'%self.n_agent)
        else:
            # when testing, var_scope comes from main learner policy (actor) network
            var_scope = ('learner_actor_main')

        # Create policy (actor) network
        if train_params.USE_BATCH_NORM:
            self.actor = Actor_BN(
                train_params.STATE_DIMS, train_params.ACTION_DIMS,\
                train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH,\
                train_params.DENSE1_SIZE, train_params.DENSE2_SIZE,\
                train_params.FINAL_LAYER_INIT, is_training=False)
        else:
            self.actor = Actor(
                train_params.STATE_DIMS, train_params.ACTION_DIMS,\
                train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH,\
                train_params.DENSE1_SIZE, train_params.DENSE2_SIZE,\
                train_params.FINAL_LAYER_INIT)

        state_size = (train_params.BATCH_SIZE, train_params.STATE_DIMS[0])
        print('AGENT BUILD STATE_SIZE: {}'.format(state_size))
        self.actor.build(state_size)


    def set_agent_weights(self, num_eps):
        # Update agent's policy network params from learner
        learner_policy_params = self.learner.actor.get_weights()
        self.actor.set_weights(learner_policy_params)


    @tf.function
    def get_action(self, state):
        # remove batch dimension from single action output
        action = self.actor(state)[0]

        # add gaussian noise
        action_bounds = (self.action_bound_high - self.action_bound_low) / 2
        noise = tf.random.normal(shape = action.shape) * action_bounds * self.noise_std
        action += (noise * self.noise_decay**self.num_eps)

        # clip to upper and lower action limits
        action = tf.clip_by_value(action,
                                  self.action_bound_low,
                                  self.action_bound_high)

        return action


    def build_summaries(self, logdir):
        # Create summary writer to write summaries to disk
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.summary_writer = tf.summary.create_file_writer(logdir)

        # Create summary op to save episode reward to Tensorboard log
        self.ep_reward_var = tf.Variable(
                0.0, trainable=False, name=('ep_reward_agent_%02d'%self.n_agent))


    def write_summary(self, episode_reward, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(name = str(episode_reward), data = self.ep_reward_var,\
                              step = step)


    def add_to_PER(self):
        state_0, action_0, reward_0, next_state_0, terminal_0 = self.exp_buffer.popleft()
        reward = reward_0
        gamma = train_params.DISCOUNT_RATE
        if 'OSMC' not in train_params.MODE:  # must discount rewards for N step case
            for (_, _, r_i) in self.exp_buffer:
                reward += r_i * gamma  # reward is discounted
                gamma *= train_params.DISCOUNT_RATE
        else:
            pass  # no need for discounting rewards in single step case
        # If learner is requesting a pause (to remove samples from PER),...
        #...wait before adding more samples
        self.run_agent_event.wait()
        self.PER_memory.add(state_0, action_0, reward, next_state_0,\
                           terminal_0, gamma)


    def run(self):
        with tf.device('/device:CPU:0'):
            # Continuously run agent in environment to collect experiences and...
            #...add to replay memory

            # Initialise deque buffer to store experiences for N-step returns
            self.exp_buffer = deque()

            # Perform initial copy of params from learner to agent
            self.set_agent_weights(-1)

            # Initially set threading event to allow agent to run until told otherwise
            self.run_agent_event.set()

            self.num_eps = 0

            while not self.stop_agent_event.is_set():
                self.num_eps += 1
                # Reset environment and experience buffer
                state_o = self.env_wrapper.reset()
                state = self.env_wrapper.normalize_state(state_o)
                self.exp_buffer.clear()

                num_steps = 0
                episode_reward = 0
                ep_done = False

                rewards = [] ### DEBUG
                truthscores = [] ### DEBUG
                while not ep_done:
                    num_steps += 1
                    ## Take action and store experience
                    if train_params.RENDER:
                        self.env_wrapper.render()
                    # Add batch dimension to single state input
                    action = self.get_action(np.expand_dims(state, 0))
                    next_state, reward, terminal, loc_impact, truthscore, t_impact,\
                            hit_miss_dist, t_opt = self.env_wrapper.step(action)
                    truthscores.append(truthscore)  ### DEBUG

                    episode_reward += reward

                    next_state = self.env_wrapper.normalize_state(next_state)
                    reward = self.env_wrapper.normalize_reward(reward)
                    rewards.append(reward)  ### DEBUG

                    self.exp_buffer.append((state, action, reward, next_state, terminal))

                    if 'OSMC' not in train_params.MODE: # must discount rewards for N step case
                        # Need at least N steps in experience buffer before we can compute...
                        #...Bellman rewards and add an N-step experience to replay memory
                        if len(self.exp_buffer) >= train_params.N_STEP_RETURNS:
                            self.add_to_PER()
                    else: # no need for discounting rewards in single step case
                        self.add_to_PER()


                    state = next_state

                    if terminal or num_steps == train_params.MAX_EP_LENGTH:
                        # Log total episode reward
                        if train_params.LOG_DIR is not None:
                            self.write_summary(episode_reward, self.num_eps)
                        # Compute Bellman rewards and add experiences to replay memory for...
                        #...last N-1 experiences still remaining in the experience buffer
                        while len(self.exp_buffer) != 0:
                            self.add_to_PER()

                        # Start next episode
                        ep_done = True


                rewardsmean = np.mean(rewards, axis = 0)  ### DEBUG
                truthscoresmean = np.mean(truthscores, axis = 0)  ### DEBUG
                # Update agent networks with learner params every 'update_agent_ep' episodes
                if self.num_eps % train_params.UPDATE_AGENT_EP == 0:
                    self.set_agent_weights(self.num_eps)
                    if self.num_eps % 1000 == 0:
                        timediff = time.time() - self.start_time
                        az = np.rad2deg(np.arctan(state_o[-1] / state_o[0]))
                        rgstate = np.linalg.norm(state_o)
                        rgactual = t_impact * self.env_wrapper.vel * np.cos(action[-1])
                        # print()
                        # print('\nAGENT: ' +\
                        #       '{} n_ep: {}, hit_miss_dist: {:.2e}, t_truth: {:.2e}, rgstate: {:.2f}, rgactual: {:.2f}, action: [{:.2f}], rewards: [{:.3f}], time: {:.2f}'.\
                        #     format(self.n_agent, self.num_eps,
                        #            hit_miss_dist, truthscoresmean[1], rgstate, rgactual,
                        #            np.rad2deg(action[0]), reward[0], timediff))

                        # print('\nSTATE: {}'.format(state_o))
                        # print('LOC_IMPACT: {}'.format(loc_impact))
                        # print('imact2tgt: {}'.format(np.subtract(loc_impact, state_o)))
                        print('\nAGENT: ' +\
                              '{} n_ep: {}, hit_miss_dist: {:.2e}, t_truth: {:.2e}, az: {:.2f}, rgstate: {:.2f}, rgactual: {:.2f}, action: [{:.2f} {:.2f}], rewards: [{:.3f} {:.3f}], time: {:.2f}'.\
                            format(self.n_agent, self.num_eps,
                                   hit_miss_dist, truthscoresmean[1], az, rgstate, rgactual,
                                   np.rad2deg(action[0]), np.rad2deg(action[-1]), reward[0], reward[-1], timediff))
                    self.learner.agent_rewards = '{:.3e}'.format(hit_miss_dist)


    def test(self):
        # Test a saved ckpt of actor network and save results to file (optional)

        def load_ckpt(ckpt_dir, ckpt_file):
            ckpt_pth = ckpt_dir + '/' + ckpt_file
            print(ckpt_pth)
            self.actor.load_weights(ckpt_pth)
            sys.stdout.write('%s restored.\n\n' % ckpt_pth)
            sys.stdout.flush()

            self.train_ep = int(ckpt_pth.split('_')[2].split('-')[0])

        # Load ckpt from ckpt_dir
        load_ckpt(test_params.CKPT_DIR, test_params.CKPT_FILE)

        # Create Tensorboard summaries to save episode rewards
        if test_params.LOG_DIR is not None:
            self.build_summaries(test_params.LOG_DIR)

        rewards = []

        for test_ep in range(1, test_params.NUM_EPS_TEST+1):
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalize_state(state)
            ep_reward = 0
            step = 0
            ep_done = False

            while not ep_done:
                if test_params.RENDER:
                    self.env_wrapper.render()
                # Add batch dimension to single state input
                action = self.get_action(np.expand_dims(state, 0))  # .numpy()
                state, reward, terminal = self.env_wrapper.step(action)
                state = self.env_wrapper.normalize_state(state)

                ep_reward += reward
                step += 1

                # Episode can finish either by reaching terminal state or max episode steps
                if terminal or step == test_params.MAX_EP_LENGTH:
                    sys.stdout.write('\x1b[2K\rTest episode {:d}/{:d}'.\
                            format(test_ep, test_params.NUM_EPS_TEST))
                    sys.stdout.flush()
                    rewards.append(ep_reward)
                    ep_done = True

        mean_reward = np.mean(rewards)
        error_reward = ss.sem(rewards)

        sys.stdout.write(
                '\x1b[2K\rTesting complete \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.\
                format(mean_reward, error_reward))
        sys.stdout.flush()

        # Log average episode reward for Tensorboard visualisation
        if test_params.LOG_DIR is not None:
            self.write_summary(self.train_ep, self.train_ep)

        # Write results to file
        if test_params.RESULTS_DIR is not None:
            if not os.path.exists(test_params.RESULTS_DIR):
                os.makedirs(test_params.RESULTS_DIR)
            output_file = open(test_params.RESULTS_DIR + '/' + test_params.ENV + '.txt' , 'a')
            output_file.write('Training Episode {}: \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(self.train_ep, mean_reward, error_reward))
            output_file.flush()
            sys.stdout.write('Results saved to file \n\n')
            sys.stdout.flush()

        self.env_wrapper.close()

    def play(self):
        # Play a saved ckpt of actor network in the environment, visualise performance on screen and save a GIF (optional)

        def load_ckpt(ckpt_dir, ckpt_file):
            # Load ckpt given by ckpt_file, or else load latest ckpt in ckpt_dir
            loader = tf.compat.v1.train.Saver()
            if ckpt_file is not None:
                ckpt = ckpt_dir + '/' + ckpt_file
            else:
                ckpt = tf.train.latest_checkpoint(ckpt_dir)

            loader.restore(self.sess, ckpt)
            sys.stdout.write('%s restored.\n\n' % ckpt)
            sys.stdout.flush()

            ckpt_split = ckpt.split('-')
            self.train_ep = ckpt_split[-1]

        # Load ckpt from ckpt_dir
        load_ckpt(play_params.CKPT_DIR, play_params.CKPT_FILE)

        # Create record directory
        if not os.path.exists(play_params.RECORD_DIR):
            os.makedirs(play_params.RECORD_DIR)

        for ep in range(1, play_params.NUM_EPS_PLAY+1):
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalize_state(state)
            step = 0
            ep_done = False

            while not ep_done:
                frame = self.env_wrapper.render()
                if play_params.RECORD_DIR is not None:
                    filepath = play_params.RECORD_DIR + '/Ep%03d_Step%04d.jpg' % (ep, step)
                    cv2.imwrite(filepath, frame)
                action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                state, _, terminal = self.env_wrapper.step(action)
                state = self.env_wrapper.normalize_state(state)

                step += 1

                # Episode can finish either by reaching terminal state or max episode steps
                if terminal or step == play_params.MAX_EP_LENGTH:
                    ep_done = True

        # Convert saved frames to gif
        if play_params.RECORD_DIR is not None:
            images = []
            for file in sorted(os.listdir(play_params.RECORD_DIR)):
                # Load image
                filename = play_params.RECORD_DIR + '/' + file
                if filename.split('.')[-1] == 'jpg':
                    im = cv2.imread(filename)
                    images.append(im)
                    # Delete static image once loaded
                    os.remove(filename)

            # Save as gif
            imageio.mimsave(play_params.RECORD_DIR + '/%s.gif' % play_params.ENV, images, duration=0.01)

        self.env_wrapper.close()

