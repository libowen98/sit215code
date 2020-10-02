#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
import gym
import random
import numpy as np

from collections import deque
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

Episode_reward = []
Loss = []
class DQN:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if os.path.exists('dqn.h5'):
            self.model.load_weights('dqn.h5')

        # Experience pool
        self.memory_buffer = deque(maxlen=2000)
        # Q_value to calculate the discount rate of the future reward
        self.gamma = 0.95
        # The degree to which the method of greedy choice randomly selects
        self.epsilon = 1.0
        # The decay rate of the above parameters
        self.epsilon_decay = 0.995
        # Minimum probability of random exploration
        self.epsilon_min = 0.01

        self.env = gym.make('CartPole-v0')

    def build_model(self):
        #Basic network structure
        inputs = Input(shape=(4,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        x = Dense(2, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        return model

    def update_target_model(self):
        """updata target_model
        """
        self.target_model.set_weights(self.model.get_weights())

    def egreedy_action(self, state):
        """
        Arguments:
            state
        Returns:
            action
        """
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 1)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Add data to the experience pool
        Arguments:
            state
            action
            reward
            next_state
            done
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        """update epsilon
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch):
        """
        Arguments:
            batch: batch size
        Returns:
            X: states
            y: [Q_value1, Q_value2]
        """
        # Randomly sampling a batch from the experience pool
        data = random.sample(self.memory_buffer, batch)
        # Generate Q_target。
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])

        y = self.model.predict(states)
        q = self.target_model.predict(next_states)
        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target

        return states, y

    def train(self, episode, batch):
        """
        Arguments:
            episode: game times
            batch： batch size

        Returns:
            history: Training records
        """
        self.model.compile(loss='mse', optimizer=Adam(1e-3))

        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        count = 0
        for i in range(episode):
            observation = self.env.reset()
            reward_sum = 0
            loss = np.infty
            done = False

            while not done:
                # Select the action through the greedy choice method ε-greedy
                x = observation.reshape(-1, 4)
                action = self.egreedy_action(x)
                observation, reward, done, _ = self.env.step(action)
                # Add the data to the experience pool
                reward_sum += reward
                self.remember(x[0], action, reward, observation, done)

                if len(self.memory_buffer) > batch:
                    # Train
                    X, y = self.process_batch(batch)
                    loss = self.model.train_on_batch(X, y)

                    count += 1
                    # Reduce the epsilon parameter of the egreedy.
                    self.update_epsilon()

                    # The target_Model is updated a fixed number of times
                    if count != 0 and count % 20 == 0:
                        self.update_target_model()

            Episode_reward.append(reward_sum)

            if i % 5 == 0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)

                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i+1, reward_sum, loss,
                                                                                          self.epsilon))

        self.model.save_weights('dqn.h5')

        return history

    def play(self):
        """Test the game with a trained model
        """
        observation = self.env.reset()
        count = 0
        reward_sum = 0
        random_episodes = 0

        while random_episodes < 10:
            self.env.render()
            x = observation.reshape(-1, 4)
            q_values = self.model.predict(x)[0]
            action = np.argmax(q_values)
            observation, reward, done, _ = self.env.step(action)
            count += 1
            reward_sum += reward

            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
                random_episodes += 1
                reward_sum = 0
                count = 0
                observation = self.env.reset()

        self.env.close()


if __name__ == '__main__':
    model = DQN()
    history = model.train(500, 32)
    model.play()
    plt.plot(Episode_reward)
    plt.title('Episode_reward(DQN)')
    plt.xlabel('Episode')
    plt.show()


