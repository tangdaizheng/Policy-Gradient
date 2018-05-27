#!/usr/bin/python3
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


class PolicyGradient(object):
    """
    PG类
    """

    def __init__(self, n_actions, n_features, lr=0.01, gamma=0.95, output_graph=False):
        """
        init
        :param n_actions:
        :param n_features:
        :param lr: learning rate
        :param gamma: reward decay
        :param output_graph:
        """

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = gamma
        self.output_graph = output_graph

        self.ep_ss, self.ep_as, self.ep_rs = [], [], []  # 每个 episode 数据

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter('logs/', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], 'states')
            self.a = tf.placeholder(tf.int32, [None, ], name='actions')
            self.r = tf.placeholder(tf.float32, [None, ], 'rewards')
        fc1 = tf.layers.dense(
            inputs=self.s,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        all_act = tf.layers.dense(
            inputs=fc1,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='all_act'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='all_act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.a)
            loss = tf.reduce_mean(neg_log_prob * self.r)

        with tf.name_scope('train'):
            self.train = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def _discount_norm_rewards(self):
        """
        标准化
        :return:
        """
        discount_rewards = np.zeros_like(self.ep_rs)
        reward = 0
        for i in reversed(range(0, len(self.ep_rs))):
            reward = reward * self.gamma + self.ep_rs[i]
            discount_rewards[i] = reward
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards)
        return discount_rewards

    def store_transition(self, s, a, r):
        self.ep_ss.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def choose_action(self, states):
        s = states[np.newaxis, :]
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.s: s})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def learn(self):
        discount_reward = self._discount_norm_rewards()

        self.sess.run(self.train, feed_dict={
            self.s: np.vstack(self.ep_ss),
            self.a: np.array(self.ep_as),
            self.r: discount_reward
        })
        self.ep_ss, self.ep_as, self.ep_rs = [], [], []
        return discount_reward





