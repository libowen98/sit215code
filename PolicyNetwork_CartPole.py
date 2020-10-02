#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import gym
import numpy as np
import matplotlib.pyplot as plt

Episode_reward = []
class PolicyGradient:
    def __init__(self, action_dim, state_dim, gamma, learning_rate):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.learning_rate = learning_rate

        self._build_network_()
        self.sess = tf.Session()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        self.sess.run(init_g)
        self.sess.run(init_l)

    def _build_network_(self):
        with tf.name_scope("input_layers"):
            self.tf_states = tf.placeholder(tf.float32, [None, self.state_dim], name="states")
            self.tf_action = tf.placeholder(tf.int32, [None, ], name="actions")
            self.tf_reward = tf.placeholder(tf.float32, [None, ], name="rewards")
        hidden_layer = tf.layers.dense(inputs=self.tf_states,
                                       units=10,
                                       activation=tf.nn.tanh,
                                       use_bias=True,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                       bias_initializer=tf.constant_initializer(0.1),
                                       name="hidden_layer")
        output_act = tf.layers.dense(inputs=hidden_layer,
                                     units=self.action_dim,
                                     activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0,
                                                                                     stddev=0.3),
                                     bias_initializer=tf.constant_initializer(0.1),
                                     name="output_act")
        self.output_act_probability = tf.nn.softmax(output_act, name="output_act_probability")

        with tf.name_scope("loss"):
            temp = tf.reduce_sum(-tf.log(self.output_act_probability) * tf.one_hot(self.tf_action, self.action_dim),
                                 axis=1)
            loss = tf.reduce_mean(temp * self.tf_reward)
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def choose_action(self, state):
        act_probability = self.sess.run(self.output_act_probability, feed_dict={self.tf_states: state[np.newaxis, :]})
        return np.random.choice(range(act_probability.shape[1]), p=act_probability.ravel())

    def _discount_norm_reward(self, reward_list):
        discount_norm_reward = np.zeros_like(reward_list)
        running_add = 0
        for i in reversed(range(0, len(reward_list))):
            running_add = self.gamma * running_add + reward_list[i]
            discount_norm_reward[i] = running_add
        discount_norm_reward -= np.mean(discount_norm_reward)
        discount_norm_reward /= np.std(discount_norm_reward)
        return discount_norm_reward

    def train(self, state_list, action_list, reward_list):
        discount_norm_reward = self._discount_norm_reward(reward_list)
        self.sess.run(self.optimizer, feed_dict={self.tf_states: np.array(state_list),
                                                 self.tf_action: np.array(action_list),
                                                 self.tf_reward: discount_norm_reward})

pl_brain = PolicyGradient(
    action_dim=2,
    state_dim=4,
    gamma=0.99,
    learning_rate=0.02
)

#run a episode
def make_episode():
    state_list, action_list, reward_list = [], [], []
    env = gym.make("CartPole-v0")
    state = env.reset()
    reward_sum = 0
    while True:
        # env.render()
        state_list.append(state)

        action = pl_brain.choose_action(state)
        action_list.append(action)

        state, reward, done, info = env.step(action)
        reward_list.append(reward)

        reward_sum += reward

        if done:
            break
    return state_list, action_list, reward_list, reward_sum

if __name__ == '__main__':

    for i in range(0, 500):
        state_list, action_list, reward_list, reward_sum = make_episode()
        print("Episode {}, Reward Sum {}.".format(i, reward_sum))
        Episode_reward.append(reward_sum)
        pl_brain.train(state_list, action_list, reward_list)

    plt.plot(Episode_reward)
    plt.title('Episode_reward(Policy Gradient)')
    plt.xlabel('Episode')
    plt.show()
