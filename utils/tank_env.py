'''
A custom tank battle environment.

From a fixed position, determine the turret azimuth and elevation to hit a fixed target.

state: <target range, target cross range>
action: <turret az, turret el>
reward: <abs(range miss dist), abs(cross range miss dist)>
'''


import numpy as np
from scipy.optimize import fsolve

class Tank_Env():
    def __init__(self, mode, seed = None):
        np.random.seed(seed)
        self.mode = mode

        self.vel = 100
        self.grav = -9.81

        self.elmin = np.deg2rad(1)
        velrgdistmin = self.vel * np.cos(self.elmin)
        velhtdistmin = self.vel * np.sin(self.elmin)
        t_distmin = np.roots([0.5 * self.grav, velhtdistmin, 0])[0]
        distmin = velrgdistmin * t_distmin

        self.elopt = np.deg2rad(45)
        velrgopt = self.vel * np.cos(self.elopt)
        velhtopt = self.vel * np.sin(self.elopt)
        self.t_distmax = np.roots([0.5 * self.grav, velhtopt, 0])[0]
        distmax = velrgopt * self.t_distmax

        self.elmax = np.deg2rad(89)
        velhtmax = self.vel * np.sin(self.elmax)
        self.t_max = np.roots([0.5 * self.grav, velhtmax, 0])[0]

        if '2D' in self.mode:
            self.azmin = -np.pi / 4
            self.azmax = np.pi / 4
        elif '1D' in self.mode:
            self.azmin = 0
            self.azmax = 0
        self.hitthresh = 10

        self.drangemin = distmin * np.cos(self.azmax)
        self.drangemax = distmax
        self.xrangemin = distmax * np.sin(self.azmin)
        self.xrangemax = distmax * np.sin(self.azmax)

        if '2D' in self.mode:
            self.stateshape = (2,)
            self.actionshape = (2,)
        elif '1D' in self.mode:
            self.stateshape = (1,)
            self.actionshape = (1,)
        self.state = np.zeros(self.stateshape)
        self.action = np.zeros(self.actionshape)

        if '2D' in self.mode:
            self.statemin = np.array([self.drangemin, self.xrangemin])
            self.statemax = np.array([self.drangemax, self.xrangemax])
            self.actionmin = np.array([self.azmin, self.elmin])
            self.actionmax = np.array([self.azmax, self.elopt])
        elif '1D' in self.mode:
            self.statemin = np.array([self.drangemin])
            self.statemax = np.array([self.drangemax])
            self.actionmin = np.array([self.elmin])
            self.actionmax = np.array([self.elopt])

        if '2crit' in self.mode:
            self.rewardmax = np.array([self.drangemax - self.drangemin,
                                      self.xrangemax - self.xrangemin])
        elif '1crit' in self.mode:
            self.rewardmax = np.array([self.drangemax - self.drangemin])

        self.v_min = -1.0
        self.v_max = 0.0


    def get_state_dims(self):
        return self.stateshape


    def get_action_dims(self):
        return self.actionshape


    def get_state_bounds(self):
        statebounds = [self.statemin, self.statemax]

        return statebounds


    def get_action_bounds(self):
        actionbounds = [self.actionmin, self.actionmax]

        return actionbounds


    # def set_random_seed(self, seed):
    #     np.random.seed(seed)


    def reset(self):
        if '2D' in self.mode:
            az = np.random.uniform(self.azmin, self.azmax)
        elif '1D' in self.mode:
            az = 0
        el = np.random.uniform(self.elmin, self.elmax)

        velrgdist = self.vel * np.cos(el)
        velhtdist = self.vel * np.sin(el)
        t_dist = np.roots([0.5 * self.grav, velhtdist, 0])[0]
        dist = velrgdist * t_dist
        d_range = dist * np.cos(az)
        if '2D' in self.mode:
            x_range = dist * np.sin(az)
            self.state = np.array([d_range, x_range])
        elif '1D' in self.mode:
            self.state = np.array([d_range])

        self.dist = np.linalg.norm(self.state)

        return self.state


    def normalize_state(self, state):
        normalized_state = np.divide(state, self.statemax)

        return normalized_state


    def normalize_reward(self, reward):
        normalized_reward = np.divide(reward, self.rewardmax)

        return normalized_reward


    def compute_reward(self):
        impact2tgt = np.subtract(self.loc_impact, self.state)
        self.hit_miss_dist = np.linalg.norm(impact2tgt)
        # treward = self.t_impact * np.power(self.hit_miss_dist, 2) / 3e3
        # reward = -np.array([self.hit_miss_dist, treward])
        if '2crit' in self.mode:
            reward = -np.abs(impact2tgt)
        elif '1crit' in self.mode:
            reward = -np.array([self.hit_miss_dist])

        return reward


    def compute_truthscore(self):  # accessing from outside
        def t_hit_opt(x):
            z = x - self.dist /\
                (self.vel * np.cos(np.arcsin(-0.5 * self.grav * x / self.vel)))

            return z

        t_opt = fsolve(t_hit_opt, self.t_distmax / 2)[0]
        tdiff = np.abs(self.t_impact - t_opt)
        truthscore = np.array([self.hit_miss_dist, tdiff])

        return truthscore, t_opt


    def step(self, action):
        if '2D' in self.mode:
            az = action[0]
        elif '1D' in self.mode:
            az = 0
        el = action[-1]

        velht = self.vel * np.sin(el)
        veldist = self.vel * np.cos(el)
        velrg = veldist * np.cos(az)

        self.t_impact = np.roots([0.5 * self.grav, velht, 0])[0]

        rg_impact = velrg * self.t_impact
        if '2D' in self.mode:
            xr_impact = veldist * np.sin(az) * self.t_impact
            self.loc_impact = np.array([rg_impact, xr_impact])
        elif '1D' in self.mode:
            self.loc_impact = np.array([rg_impact])

        reward = self.compute_reward()
        truthscore, t_opt = self.compute_truthscore()

        next_state = self.reset()

        return next_state, reward, True, self.loc_impact, truthscore, self.t_impact, self.hit_miss_dist, t_opt


    # def set_state(self, state):
    #     self.state = state
    #     self.dist = np.linalg.norm(state)

    #     return self.state
