import numpy as np

import matplotlib.style

import matplotlib
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')

"""
Contains the definition of the agent that will run in an
environment.
"""

ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
ACT_TORCH_UP    = 5
ACT_TORCH_DOWN  = 6
ACT_TORCH_LEFT  = 7
ACT_TORCH_RIGHT = 8






class RandomAgent:

    def __init__(self,games):
        """
            learning rate value
            gamma value
        """
        self.learning_rate = .1   # learning rate
        self.gamma = .8

        self.epsilon_a = -1/90

        # State-action function (Q-function)
        self.q_table = {}
        

        # Episode is defined as each time we starting moving until we die
        self.n_episode = 0

        # Step is defined as the iteration in each game
        self.step = 0

        self.cumul_reward = 0

        self.EpisodeStats = namedtuple("Stats", ["episode_rewards"])

        self.stats = self.EpisodeStats(
            episode_rewards=np.zeros(games+1))

    def plot_episode_stats(self,stats, smoothing_window=10):
        # Plot the game reward over time
        fig2 = plt.figure(figsize=(10, 5))
        rewards_smoothed = pd.Series(self.stats.episode_rewards).rolling(smoothing_window,
                                                                    min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed)
        plt.xlabel("Game")
        plt.ylabel("Game Reward")
        plt.title("Game Reward over Time (Smoothed over window size {})".format(smoothing_window))
        plt.show(fig2)
        plt.close()

    def reset(self):
        self.step = 0
        self.n_episode += 1

    def act(self, observation):
        self.step += 1
        self.epsilon = 0.5

        position, smell, breeze, charges = observation

        self.create_state(observation)
        state_action = self.q_table[observation]
        if self.n_episode >= 90:

            if smell and charges > 0:
                return self.choose_action(state_action)
            else:
                return self.choose_action(state_action[0:4])

        if np.random.uniform() >= self.epsilon_a * self.epsilon + 1:
            if smell and charges > 0:
                return self.choose_action(state_action)
            else:
                return self.choose_action(state_action[0:4])
        else:
            if smell and charges > 0:
                return np.random.randint(1,9)
            else:
                return np.random.randint(1,5)

    def reward(self, observation, action, reward,t):
        if self.step > 1:
            self.learn(s=observation,
                    a = action - 1,
                    r = reward,
                    s_= self.next_observation(observation, action),
                )
        self.stats.episode_rewards[self.n_episode] += reward
        if self.n_episode >= 90:
            self.cumul_reward += reward
        return self.q_table,self.stats

    def create_state(self, state):
        """
            If new state, create it in the q_table, with default values = 0
        """
        if state not in self.q_table.keys():
            self.q_table[state] = 8 * [10]
    
    def learn(self, s, a, r, s_):
        self.create_state(s_) # create only if not exists
        q_predict = self.q_table[s][a]
        q_target = r + self.gamma * max(self.q_table[s_])
        self.q_table[s][a] += self.learning_rate * (q_target - q_predict)



    def choose_action(self, state_action):
        # get the list of actions with maximum value in Q table
        action_with_max_value = [i for i in range(len(state_action)) if state_action[i] == max(state_action)]
        return np.random.choice(action_with_max_value) + 1

    def next_observation(self, observation, action):
        position, smell, breeze, charges = observation
        if action >= 4 and charges > 0:
            charges -= 1
            next_observation = (self.next_position(position, action), smell, breeze, charges)
        else:
            next_observation = (self.next_position(position, action), smell, breeze, charges)
        
        return next_observation

    def next_position(self, position, action):
        x, y = position
        if action == 1 and y <= 2:
            y += 1
        elif action == 2 and y >= 1:
            y -= 1
        elif action == 3 and x >= 1:
            x -= 1
        elif action == 4 and x <= 2:
            x += 1
        return x, y
            

Agent = RandomAgent
