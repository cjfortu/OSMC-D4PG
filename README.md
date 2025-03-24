# One Step Multi Critic Distributed Distributional Deep Deterministic Policy Gradient (OSMC-D4PG)
# One Step Multi Critic Deep Deterministic Policy Gradient (OSMC-DDPG)

author: Clemente Fortuna (clemente.fortuna@mtsi-va.com)

Written in Tensorflow 2.10

Multi-threaded approach based on https://github.com/msinto93/D4PG
Distributional RL based on: https://arxiv.org/pdf/1707.06887
One Step Two Critic DDPG based on: https://arxiv.org/pdf/2203.16289

PER buffer copied from: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
Segment Tree copied from: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
L2 Projection copied from: https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py

N-step D4PG and DDPG trained on Gym environments (Pendulum)
1-step DDPG trained on custom tank battle environment

**TODO: adapt distributional learning for 1-step RL**