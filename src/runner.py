"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""

class Runner:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent


    def step(self,i):
        observation = self.environment.observe()
        action = self.agent.act(observation)
        (reward, stop) = self.environment.act(action)
        Q, stats =self.agent.reward(observation, action, reward,i)
        return (observation, action, reward, stop,Q, stats)

    def loop(self, games, max_iter):
        cumul_reward = 0.0
        stats=0
        for g in range(1, games+1):
            print ("Game Number {}:".format(g))
            curr_reward = 0.0
            self.agent.reset()
            self.environment.reset() 
            for i in range(1, max_iter+1):
                print ("Iteration Number {}:".format(i))
                self.environment.display()
                (obs, act, rew, stop,Q, stats) = self.step(i)
                cumul_reward += rew
                curr_reward  += rew

                print (" reward: {}".format(rew))
                print (" current game reward: {}".format(curr_reward))
                if stop is not None:
                    print ("Terminal event: {}".format(stop))
                if stop is not None:
                    break
            print ("Finished game number: {} ".format(g))
            print ("cumulative reward: {}".format(cumul_reward))
            print ("current game Final reward: {}".format(curr_reward))
            self.environment.display()
        self.agent.plot_episode_stats(stats)
        return cumul_reward
