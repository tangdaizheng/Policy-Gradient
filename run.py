#/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from policy_gradient import PolicyGradient

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float('display_threshold', 10, 'the reward threshold to display render')
tf.flags.DEFINE_boolean('render', False, 'render waste time')
tf.flags.DEFINE_boolean('output_graph', False, 'whether to save graph')
tf.flags.DEFINE_string('env_name', 'CartPole-v0', 'env name')
tf.flags.DEFINE_integer('episode', 1000, 'train episode')

RENDER = FLAGS.render

env = gym.make(FLAGS.env_name)
env.seed(1)
env = env.unwrapped

PG = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    lr=0.02,
    gamma=0.99,
    output_graph=FLAGS.output_graph
)

for i in range(FLAGS.episode):
    s = env.reset()

    while True:
        if RENDER:
            env.render()
        action = PG.choose_action(s)
        s_, r, done, info = env.step(action)
        PG.store_transition(s_, action, r)
        if done:
            episode_rs_sum = sum(PG.ep_rs)
            if 'running_reward' not in globals():
                running_reward = episode_rs_sum
            else:
                running_reward = running_reward * 0.99 + episode_rs_sum * 0.01
            if running_reward > FLAGS.display_threshold:
                RENDER = True
            print('episode:', i, ' reward:', running_reward)

            norm_reward = PG.learn()

            if i == 30:
                plt.plot(norm_reward)
                plt.xlabel('episode steps')
                plt.ylabel('normalized reward')
                plt.show()
            break

        s = s_





