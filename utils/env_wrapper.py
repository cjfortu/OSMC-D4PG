'''
## Env Wrapper ##
# A wrapper around the OpenAI Gym environments. Each environment requires its own custom wrapper as the preprocessing required differs by environment.
@author: Mark Sinton (msinto93@gmail.com)
'''

import gym

class EnvWrapper:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)

    def reset(self):
        state = self.env.reset()
        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state, reward, terminal

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render(mode='rgb_array')
        return frame

    def get_state_dims(self):
        return self.env.observation_space.shape

    def get_state_bounds(self):
        return self.env.observation_space.low, self.env.observation_space.high

    def get_action_dims(self):
        return self.env.action_space.shape

    def get_action_bounds(self):
        return self.env.action_space.low, self.env.action_space.high

    def close(self):
        self.env.close()


class PendulumWrapper(EnvWrapper):
    def __init__(self):

        EnvWrapper.__init__(self, 'Pendulum-v0')

        # State
        # Type: Box(3)
        # Num    Observation    Min    Max
        #  0     cos(theta)    -1.0    1.0
        #  1     sin(theta)    -1.0    1.0
        #  2     theta dot     -8.0    8.0

        # Action
        # Type: Box(1)
        # Num    Action           Min    Max
        #  0     Joint effort    -2.0    2.0

        # Reward = -(theta^2 + 0.1*theta_dot^2 + 0.001*action^2)
        # Theta lies between -pi and pi
        # Goal is to remain at zero angle (theta=0), with least rotational velocity (theta_dot=0) and least effort (action=0).
        # Max reward is therefore 0.0.
        # Min reward occurs at max angle (theta_max), max rotational velocity (theta_dot_max) and max effort (action_max) - approx -16.27

        # Lower and upper bounds of critic value output distribution, these will vary with environment
        # V_min and V_max should be chosen based on the range of normalised reward values in the chosen env
        self.v_min = -20.0
        self.v_max = 0.0

    def normalize_state(self, state):
        # Normalise state values to [-1, 1] range
        return state/self.env.observation_space.high

    def normalize_reward(self, reward):
        # Normalise reward values
        return reward/100.0

class LunarLanderContinuousWrapper(EnvWrapper):
    def __init__(self):

        EnvWrapper.__init__(self, 'LunarLanderContinuous-v2')

        # Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
        # Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
        # If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points.
        # Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points.
        # Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
        # Action is two real values vector from -1 to +1. First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
        # Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.

        # Lower and upper bounds of critic value output distribution, these will vary with environment
        # V_min and V_max should be chosen based on the range of normalised reward values in the chosen env
        self.v_min = -20.0
        self.v_max = 20.0

    def normalise_state(self, state):
        # State does not need to be normalised for this env
        return state

    def normalise_reward(self, reward):
        # Normalise reward values
        return reward/10.0

class BipedalWalkerWrapper(EnvWrapper):
    def __init__(self, hardcore=False):

        if hardcore:
            EnvWrapper.__init__(self, 'BipedalWalkerHardcore-v2')
        else:
            EnvWrapper.__init__(self, 'BipedalWalker-v2')

        # Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100.
        # Applying motor torque costs a small amount of points, more optimal agent will get better score.
        # State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements.
        # There's no coordinates in the state vector.

        # Hardcore version contains ladders, stumps, pitfalls. Time limit is increased due to obstacles.

        # Lower and upper bounds of critic value output distribution, these will vary with environment
        # V_min and V_max should be chosen based on the range of normalised reward values in the chosen env
        self.v_min = -40.0
        self.v_max = 40.0

    def normalise_state(self, state):
        # State does not need to be normalised for this env
        return state

    def normalise_reward(self, reward):
        # Normalise reward values
        return reward/10.0



