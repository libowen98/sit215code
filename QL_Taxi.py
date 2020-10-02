#!/usr/bin/env python 
# -*- coding:utf-8 -*-

''' Q-learning to solve Taxi problem '''

"""
The agent must pick up the passengers in one location and drop them in another. 
If you successfully drop the passenger, the intelligence will be rewarded with
+20 points and -1 points for each time step. If the agent picks up and puts 
down incorrectly, it gains -10 points. Therefore, the goal of an agent is to 
learn to pick up and drop passengers in the right place in the shortest time 
possible, without picking up illegal passengers.
            +---------+
            |R: | : :G|
            | : : : : |
            | : : : : |
            | | : | : |
            |Y| : |B: |
            +---------+
The letters (R, G, Y, B) represent four different positions
"""

import random
import gym
import matplotlib.pyplot as plt
env = gym.make('Taxi-v3')
env.render()  # Output Taxi-v3 environment

# Now, we initialize, Q table as a dictionary which stores state-action
# pair specifying value of performing an action a in state s.
q = {}
Episode_reward = []
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s, a)] = 0.0

# We define a function called update_q_table which will update the Q values
# according to our Q learning update rule.

# If you look at the below function, we take the value which has maximum value
# for a state-action pair and store it in a variable called qa, then we update
# the Q value of the preivous state by our update rule.
def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
    qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
    q[(prev_state,action)] += alpha * (reward + gamma * qa - q[(prev_state,action)])

# Then, we define a function for performing epsilon-greedy policy. In epsilon-greedy policy,
# either we select best action with probability 1-epsilon or we explore new action with probability epsilon.
def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: q[(state,x)])

# Now we initialize necessary variables
# alpha - TD learning rate
# gamma - discount factor
# epsilon - epsilon value in epsilon greedy policy
alpha = 0.4
gamma = 0.999
epsilon = 0.017

if __name__ == "__main__":
    for i in range(500):
        reward_sum = 0
        prev_state = env.reset()

        while True:
            # env.render()

            # In each state, we select the action by epsilon-greedy policy
            action = epsilon_greedy_policy(prev_state, epsilon)

            # then we perform the action and move to the next state, and receive the reward
            nextstate, reward, done, _ = env.step(action)

            # Next we update the Q value using our update_q_table function
            # which updates the Q value by Q learning update rule
            update_q_table(prev_state, action, reward, nextstate, alpha, gamma)

            # Finally we update the previous state as next state
            prev_state = nextstate

            # Store all the rewards obtained
            reward_sum += reward

            # we will break the loop, if we are at the terminal state of the episode
            if done:
                break

        Episode_reward.append(reward_sum)
        print("[Episode %d] Total reward: %d" % (i + 1, reward_sum))
    env.close()
    plt.plot(Episode_reward)
    plt.xlabel('Episode')
    plt.show()



