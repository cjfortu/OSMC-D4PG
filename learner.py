'''
## Learner ##
Learner class - this trains the D4PG or DDPG networks on experiences sampled (by priority)...
...from the PER buffer
@author: Clemente Fortuna (clemente.fortuna@mtsi-va.com)

based on https://github.com/msinto93/D4PG
'''

import os
import sys
import tensorflow as tf
import numpy as np
import sonnet as snt
import time

from params import train_params
from utils.network import Actor, Critic
from utils.l2_projection import _l2_project


class Learner:
    def __init__(self, PER_memory, run_agent_event, stop_agent_event):
        print("Initialising learner... \n\n")

        self.PER_memory = PER_memory
        self.run_agent_event = run_agent_event
        self.stop_agent_event = stop_agent_event
        self.agent_rewards = None


    def build_networks(self):

        # Create value (critic) network + target network
        self.critic_m1 = Critic(
            train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE,\
            train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT,\
            train_params.ADDED_SIZE,\
            train_params.V_MIN, train_params.V_MAX, train_params.NUM_ATOMS,\
            )
        self.critic_m1_target = Critic(
            train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE,\
            train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT,\
            train_params.ADDED_SIZE,\
            train_params.V_MIN, train_params.V_MAX, train_params.NUM_ATOMS,\
            )
        self.critic_m2 = Critic(
            train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE,\
            train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT,\
            train_params.ADDED_SIZE,\
            train_params.V_MIN, train_params.V_MAX, train_params.NUM_ATOMS,\
            )
        self.critic_m2_target = Critic(
            train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE,\
            train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT,\
            train_params.ADDED_SIZE,\
            train_params.V_MIN, train_params.V_MAX, train_params.NUM_ATOMS,\
            )

        # Create policy (actor) network + target network
        self.actor = Actor(
            train_params.STATE_DIMS, train_params.ACTION_DIMS,\
            train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH,\
            train_params.DENSE1_SIZE, train_params.DENSE2_SIZE,\
            train_params.FINAL_LAYER_INIT)
        self.actor_target = Actor(
            train_params.STATE_DIMS, train_params.ACTION_DIMS,\
            train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH,
            train_params.DENSE1_SIZE, train_params.DENSE2_SIZE,\
            train_params.FINAL_LAYER_INIT)

        action_size = (train_params.BATCH_SIZE, train_params.ACTION_DIMS[0])
        state_size = (train_params.BATCH_SIZE, train_params.STATE_DIMS[0])
        self.actor.build(state_size)
        self.actor_target.build(state_size)
        self.critic_m1.build([state_size, action_size])
        self.critic_m1_target.build([state_size, action_size])
        self.critic_m2.build([state_size, action_size])
        self.critic_m2_target.build([state_size, action_size])

        self.critic_m1_opt = tf.keras.optimizers.Adam(train_params.CRITIC_LR, amsgrad=True)
        self.critic_m2_opt = tf.keras.optimizers.Adam(train_params.CRITIC_LR, amsgrad=True)
        self.actor_opt = tf.keras.optimizers.Adam(train_params.ACTOR_LR, amsgrad=True)

        self.m1_rew_scale = tf.cast(train_params.M1_REW_SCALE, dtype = tf.float32)
        self.m2_rew_scale = tf.cast(train_params.M2_REW_SCALE, dtype = tf.float32)

        # Create saver for saving model ckpts (we only save learner network vars)
        netnames = ['critic_m1', 'critic_m1_target',
                'critic_m2', 'critic_m2_target',
                'actor', 'actor_target']
        self.ckpt_paths = []
        for netname in netnames:
            model_name = train_params.ENV + '_' + netname
            self.ckpt_paths.append(os.path.join(train_params.CKPT_DIR, model_name))

        if not os.path.exists(train_params.CKPT_DIR):
            os.makedirs(train_params.CKPT_DIR)

        self.nets = [self.critic_m1, self.critic_m1_target,
                self.critic_m2, self.critic_m2_target,
                self.actor, self.actor_target]


    def update_target(self, model_target, model_ref):
        # Create ops which update target network params with...
        #...fraction of (tau) main network params
        model_target.set_weights(
            [
                train_params.TAU * ref_weight + (1 - train_params.TAU) * target_weight
                for (target_weight, ref_weight)
                in list(zip(model_target.get_weights(), model_ref.get_weights()))
            ]
        )


    def initialise_vars(self):
        # Load ckpt file if given, otherwise initialise variables and hard copy to target networks
        if train_params.CKPT_NUM is not None:
            ckpt_step = str(train_params.CKPT_NUM)
            self.start_step = int(ckpt_step)
            for ckpt_path, net in zip(self.ckpt_paths, self.nets):
                net.load_weights(ckpt_path + '_' + ckpt_step)
        else:
            self.start_step = 0
            ## Perform hard copy (tau=1.0) of initial params to target networks
            self.actor_target.set_weights(self.actor.get_weights())
            self.critic_m1_target.set_weights(self.critic_m1.get_weights())
            self.critic_m2_target.set_weights(self.critic_m2.get_weights())


    @tf.function
    def train_OSMC_DDPG(self, states, actions, rewards, next_states, terminals,\
                       gammas, weights):
        with tf.GradientTape() as critic_m1_tape:
            # ys = rewards +\
            #         gammas * (1 - terminals) *\
            #         self.critic_target(
            #         [next_states, self.actor_target(next_states)])
            rewards_m1 = self.m1_rew_scale *\
                    tf.cast(tf.expand_dims(rewards[:, 0], axis = 1), dtype = tf.float32)
            TD_m1_errors = rewards_m1 - self.critic_m1([states, actions])
            weighted_m1_losses = tf.cast(TD_m1_errors, dtype = tf.float32) *\
                    tf.cast(weights, dtype = tf.float32)
            critic_m1_loss = tf.math.reduce_mean(
                    tf.math.square(weighted_m1_losses))
        critic_m1_grads = critic_m1_tape.gradient(critic_m1_loss,\
                self.critic_m1.trainable_variables)
        self.critic_m1_opt.apply_gradients(
                zip(critic_m1_grads, self.critic_m1.trainable_variables))
        del critic_m1_tape

        with tf.GradientTape() as critic_m2_tape:
            # ys = rewards +\
            #         gammas * (1 - terminals) *\
            #         self.critic_target(
            #         [next_states, self.actor_target(next_states)])
            rewards_m2 = self.m2_rew_scale *\
                    tf.cast(tf.expand_dims(rewards[:, 1], axis = 1), dtype = tf.float32)
            TD_m2_errors = rewards_m2 - self.critic_m2([states, actions])
            weighted_m2_losses = tf.cast(TD_m2_errors, dtype = tf.float32) *\
                    tf.cast(weights, dtype = tf.float32)
            critic_m2_loss = tf.math.reduce_mean(
                    tf.math.square(weighted_m2_losses))
        critic_m2_grads = critic_m2_tape.gradient(critic_m2_loss,\
                self.critic_m2.trainable_variables)
        self.critic_m2_opt.apply_gradients(
                zip(critic_m2_grads, self.critic_m2.trainable_variables))
        del critic_m2_tape

        with tf.GradientTape() as actor_tape:
            m1_loss = tf.math.reduce_mean(
                    self.critic_m1([states, self.actor(states)]))
            m2_loss = tf.math.reduce_mean(
                    self.critic_m2([states, self.actor(states)]))
            actor_loss = -tf.math.add(m1_loss, m2_loss)
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(
                zip(actor_grads, self.actor.trainable_variables))
        del actor_tape

        TD_errors_abs = tf.math.add(tf.math.abs(TD_m1_errors), tf.math.abs(TD_m2_errors))

        return actor_loss, TD_errors_abs


    @tf.function
    def train_OSMC_D4PG(self, states, actions, rewards, next_states, terminals,\
                       gammas, weights):
        with tf.GradientTape() as critic_m1_tape:
            rewards_m1 = rewards[:, 0]
            # start critic_m1 training
            future_action = self.actor_target(next_states)
            _, target_Z_dist, target_Z_atoms =\
                    self.critic_m1_target([next_states, future_action])
            target_Z_atoms = tf.repeat(tf.expand_dims(target_Z_atoms, axis = 0),\
                    tf.constant(train_params.BATCH_SIZE), axis = 0)
            target_Z_atoms = tf.where(
                    tf.expand_dims(terminals, axis = 1),
                    y = tf.cast(target_Z_atoms, dtype = tf.float32),
                    x = tf.zeros(target_Z_atoms.shape))
            target_Z_atoms =\
                    tf.expand_dims(tf.cast(rewards_m1, dtype = tf.float32), axis = 1) +\
                    (target_Z_atoms *\
                    tf.expand_dims(tf.cast(gammas, dtype = tf.float32), axis = 1))
            # target_Z_atoms = tf.expand_dims(tf.cast(rewards_m1, dtype = tf.float32), axis = 1)

            # enter critic_m1
            output_logits, _, z_atoms = self.critic_m1([states, actions])
            target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, z_atoms)
            TD_m1_errors = tf.nn.softmax_cross_entropy_with_logits(
                    logits = output_logits, labels = tf.stop_gradient(target_Z_projected))
            weighted_loss = tf.cast(TD_m1_errors, dtype = tf.float32) *\
                    tf.cast(weights, dtype = tf.float32)
            mean_loss = tf.reduce_mean(weighted_loss)
            l2_reg_loss = tf.math.add_n(
                    [tf.nn.l2_loss(v) for v in self.critic_m1.trainable_variables\
                    if 'kernel' in v.name]) * train_params.CRITIC_L2_LAMBDA
            total_loss = mean_loss + l2_reg_loss

        critic_grads = critic_m1_tape.gradient(total_loss, self.critic_m1.trainable_variables)
        self.critic_m1_opt.apply_gradients(
                zip(critic_grads, self.critic_m1.trainable_variables))
        del critic_m1_tape


        with tf.GradientTape() as critic_m2_tape:
            rewards_m2 = rewards[:, 1]
            # start critic_m2 training
            future_action = self.actor_target(next_states)
            _, target_Z_dist, target_Z_atoms =\
                    self.critic_m2_target([next_states, future_action])
            target_Z_atoms = tf.repeat(tf.expand_dims(target_Z_atoms, axis = 0),\
                    tf.constant(train_params.BATCH_SIZE), axis = 0)
            target_Z_atoms = tf.where(
                    tf.expand_dims(terminals, axis = 1),
                    y = tf.cast(target_Z_atoms, dtype = tf.float32),
                    x = tf.zeros(target_Z_atoms.shape))
            target_Z_atoms =\
                    tf.expand_dims(tf.cast(rewards_m2, dtype = tf.float32), axis = 1) +\
                    (target_Z_atoms *\
                    tf.expand_dims(tf.cast(gammas, dtype = tf.float32), axis = 1))
            # target_Z_atoms = tf.expand_dims(tf.cast(rewards_m2, dtype = tf.float32), axis = 1)

            # enter critic_m2
            output_logits, _, z_atoms = self.critic_m2([states, actions])
            target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, z_atoms)
            TD_m2_errors = tf.nn.softmax_cross_entropy_with_logits(
                    logits = output_logits, labels = tf.stop_gradient(target_Z_projected))
            weighted_loss = tf.cast(TD_m2_errors, dtype = tf.float32) *\
                    tf.cast(weights, dtype = tf.float32)
            mean_loss = tf.reduce_mean(weighted_loss)
            l2_reg_loss = tf.math.add_n(
                    [tf.nn.l2_loss(v) for v in self.critic_m2.trainable_variables\
                    if 'kernel' in v.name]) * train_params.CRITIC_L2_LAMBDA
            total_loss = mean_loss + l2_reg_loss

        critic_grads = critic_m2_tape.gradient(total_loss, self.critic_m2.trainable_variables)
        self.critic_m2_opt.apply_gradients(
                zip(critic_grads, self.critic_m2.trainable_variables))
        del critic_m2_tape


        with tf.GradientTape(persistent = True) as actor_tape:
            # start actor training
            actor_actions = self.actor(states)
            _, output_probs_m1, z_atoms = self.critic_m1([states, actor_actions])
            _, output_probs_m2, z_atoms = self.critic_m2([states, actor_actions])
        # action_grads= actor_tape.gradient(
        #         target = output_probs,
        #         sources = actor_actions,
        #         output_gradients = z_atoms)
        # print('LEAERNER action_grads:\n{}\n{}'.format(action_grads.shape, action_grads))
        action_grads_m1 = actor_tape.gradient(
                target = output_probs_m1,
                sources = actor_actions,
                output_gradients = z_atoms)
        # print('LEAERNER action_grads_m1:\n{}\n{}'.format(action_grads_m1.shape, action_grads_m1))
        action_grads_m2 = actor_tape.gradient(
                target = output_probs_m2,
                sources = actor_actions,
                output_gradients = z_atoms)
        # print('LEAERNER action_grads_m2:\n{}\n{}'.format(action_grads_m2.shape, action_grads_m2))
        action_grads = tf.math.add(action_grads_m1, action_grads_m2)
        # action_grads = action_grads_m1
        # print('LEAERNER action_grads:\n{}\n{}'.format(action_grads.shape, action_grads))
        # enter actor
        grads = actor_tape.gradient(
                target = actor_actions,
                sources = self.actor.trainable_variables,
                output_gradients = -action_grads)
        grads_scaled = list(map(lambda x: tf.divide(x, train_params.BATCH_SIZE), grads))
        self.actor_opt.apply_gradients(zip(grads_scaled, self.actor.trainable_variables))
        del actor_tape

        TD_errors_abs = tf.math.add(tf.math.abs(TD_m1_errors), tf.math.abs(TD_m2_errors))

        return None, TD_errors_abs


    def train_networks(self, states, actions, rewards, next_states, terminals,\
                       gammas, weights):
        if 'OSMC-DDPG' in train_params.MODE:
            actor_losses, critic_losses =\
                    self.train_OSMC_DDPG(states, actions, rewards, next_states, terminals,\
                        gammas, weights)

        elif 'OSMC-D4PG' == train_params.MODE:
            actor_losses, critic_losses =\
                    self.train_OSMC_D4PG(states, actions, rewards, next_states, terminals,\
                        gammas, weights)

        return actor_losses, critic_losses


    def run(self):
        # Sample batches of experiences from replay memory and train learner networks

        # Initialise beta to start value
        priority_beta = train_params.PRIORITY_BETA_START
        beta_increment = (train_params.PRIORITY_BETA_END -\
                          train_params.PRIORITY_BETA_START) / train_params.NUM_STEPS_TRAIN

        # Can only train when we have at least batch_size num of samples in replay memory
        while len(self.PER_memory) <= train_params.BATCH_SIZE:
            sys.stdout.write('\rPopulating replay memory up to batch_size samples...')
            sys.stdout.flush()

        # Training
        sys.stdout.write('\n\nTraining...\n')
        sys.stdout.flush()

        for train_step in range(self.start_step+1, train_params.NUM_STEPS_TRAIN+1):
            # Get minibatch
            minibatch = self.PER_memory.sample(train_params.BATCH_SIZE, priority_beta)
            states = minibatch[0]
            actions = minibatch[1]
            rewards = minibatch[2]
            next_states = minibatch[3]
            terminals = minibatch[4]
            gammas = minibatch[5]
            weights = minibatch[6]
            idxs = minibatch[7]

            # execute training
            with tf.device('/device:GPU:0'):
                actor_loss, TD_errors_abs = self.train_networks(
                        states, actions, rewards, next_states, terminals, gammas, weights)

            # update replay buffer priorities
            self.PER_memory.update_priorities(
                    idxs, (TD_errors_abs.numpy() + train_params.PRIORITY_EPSILON))

            # Update target networks weights
            if train_params.MODE == 'OSMC-D4PG' and\
                    train_step % train_params.UPDATE_TARGET_EP == 0:
                self.update_target(self.actor_target, self.actor)
                self.update_target(self.critic_m1_target, self.critic_m1)
                self.update_target(self.critic_m2_target, self.critic_m2)

            # Increment beta value at end of every step
            priority_beta += beta_increment

            # Periodically check capacity of replay mem and remove samples (by FIFO process) above this capacity
            if train_step % train_params.REPLAY_MEM_REMOVE_STEP == 0:
                if len(self.PER_memory) > train_params.REPLAY_MEM_SIZE:
                    # Prevent agent from adding new experiences to replay memory while learner removes samples
                    self.run_agent_event.clear()
                    samples_to_remove = len(self.PER_memory) - train_params.REPLAY_MEM_SIZE
                    self.PER_memory.remove(samples_to_remove)
                    # Allow agent to continue adding experiences to replay memory
                    self.run_agent_event.set()

            sys.stdout.write('\rStep {:d}/{:d}'.format(train_step, train_params.NUM_STEPS_TRAIN))
            sys.stdout.flush()

            # Save ckpt periodically
            if train_step % train_params.SAVE_CKPT_STEP == 0:
                print('REWARDS BATCH: {}'.format(np.mean(rewards)))
                print('LOSS: {}'.format(np.mean(TD_errors_abs)))

                for net, ckpt_path in zip(self.nets, self.ckpt_paths):
                    net.save_weights(ckpt_path + '_' + str(train_step) +\
                                     '{}'.format(self.agent_rewards))
                    print("Saved weights for step {}: {}".format(train_step, ckpt_path))
                sys.stdout.write('\nWeights saved.\n')
                sys.stdout.flush()
                self.agent_rewards = None

        # Stop the agents
        self.stop_agent_event.set()
