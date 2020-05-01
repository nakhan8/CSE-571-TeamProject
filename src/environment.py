from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import agent
import environment
import runner


"""
This file contains the definition of the environment
in which the agents are run.
"""

ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
ACT_TORCH_UP    = 5
ACT_TORCH_DOWN  = 6
ACT_TORCH_LEFT  = 7
ACT_TORCH_RIGHT = 8

ST_NOTHING = 0
ST_HOLE = 1

class Environment:

    def __init__(self, gridsize=(4,4), num_wumpus=1, num_holes=3, charges=3):

        self.gridsize = gridsize
        print (gridsize)
        self.num_cases = gridsize[0]*gridsize[1]
        self.num_wumpus = num_wumpus
        self.num_holes = num_holes
        self.max_charges = charges
        # intial state
        rand_idx = np.random.choice(self.num_cases, self.num_wumpus+self.num_holes+2, replace=False)
        rand_pos = [ (n%self.gridsize[1], n//self.gridsize[1]) for n in rand_idx ]
        self.init_holes = rand_pos[0:self.num_holes]
        self.init_wumpus = rand_pos[self.num_holes:-2]
        self.init_treasure = rand_pos[-2]
        self.init_agent = rand_pos[-1]

        # self.init_holes = [(0,3),(3,1),(2,0)]
        # self.init_wumpus = [(0,0)]
        # self.init_treasure = (2,3)
        # self.init_agent = (3,3)
        self.reset()

    def reset(self):
        self.rem_charges = self.max_charges
        self.wumpus = list(self.init_wumpus)
        self.holes = list(self.init_holes)
        self.treasure = self.init_treasure
        self.agent = self.init_agent

    def observe(self):
        return (self.agent, self.is_near_wumpus(), self.is_near_hole(), self.rem_charges)

    def is_near_wumpus(self):
        (ax, ay) = self.agent
        for (nx, ny) in self.wumpus:
            if (ax-nx)**2 + (ay-ny)**2 <= 1:
                return True
        return False

    def is_near_hole(self):
        (ax, ay) = self.agent
        for (nx, ny) in self.holes:
            if (ax-nx)**2 + (ay-ny)**2 <= 1:
                return True
        return False

    def move_object(self, pos, action):
        (x, y) = pos
        if action == ACT_UP:
            y += 1
        elif action == ACT_DOWN:
            y -= 1
        elif action == ACT_LEFT:
            x -= 1
        elif action == ACT_RIGHT:
            x += 1
        x = max(0, min(self.gridsize[0]-1, x))
        y = max(0, min(self.gridsize[1]-1, y))
        return (x, y)

    def kill_wumpus_at(self, pos):
        if pos in self.wumpus:
            self.wumpus.remove(pos)
            return True
        return False

    def torchlight(self, action):
        if self.rem_charges <= 0:
            return False
        (x,y) = self.agent
        if action == ACT_TORCH_UP:
            self.rem_charges -= 1
            return self.kill_wumpus_at((x, y+1))
        elif action == ACT_TORCH_DOWN:
            self.rem_charges -= 1
            return self.kill_wumpus_at((x, y-1))
        elif action == ACT_TORCH_LEFT:
            self.rem_charges -= 1
            return self.kill_wumpus_at((x-1, y))
        elif action == ACT_TORCH_RIGHT:
            self.rem_charges -= 1
            return self.kill_wumpus_at((x+1, y))
        else:
            return False

    def act(self, action):
        """Perform given action by the agent on the environment,
        and returns a reward.
        """
        self.agent = self.move_object(self.agent, action)
        if self.agent in self.holes:
            return (-10.0, "Fell into a hole.")
        if self.agent in self.wumpus:
            return (-10.0, "Encountered a Wumpus.")
        if self.torchlight(action):
            return (10.0, None)
        if self.agent == self.treasure:
            return (100.0, "Found the treasure.")
        return (-1.0, None)

    def display(self):
        print ("+-", end='')
        for x in range(self.gridsize[0]):
            print ("--", end='')
        print ("+ ")
        for y in range(self.gridsize[1]-1, -1, -1):
            print ("| ", end='')
            for x in range(self.gridsize[0]):
                if (x,y) in self.wumpus:
                    print ("W ", end='')
                elif (x, y) in self.holes:
                    print ("O ", end='')
                elif (x, y) == self.treasure:
                    print ("$ ", end='')
                elif (x, y) == self.agent:
                    print ("A ", end='')
                else:
                    print (". ", end='')
            print ("|")
        print ("+-", end='')
        for x in range(self.gridsize[0]):
            print ("--", end='')
        print ("+")




parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment', metavar='ENV_CLASS', type=str, default='Environment',
                    help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str,
                    help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--ngames', type=int, metavar='n', default='100', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='100', help='max number of iterations per game')
parser.add_argument('--gridsize', type=int, nargs="+", default=(4, 4), help='grid size')
parser.add_argument('--numwumpus', type=int, metavar='n', default='1', help='number of wumpus')
parser.add_argument('--numholes', type=int, metavar='n', default='3', help='number of holes')
parser.add_argument('--bullets', type=int, metavar='n', default='3', help='number of bullets')

def main():
    args = parser.parse_args()
    agent_class = eval('agent.{}'.format(args.agent))
    env_class = eval('environment.{}'.format(args.environment))
    my_runner = runner.Runner(env_class(args.gridsize, args.numwumpus, args.numholes, args.bullets),
                              agent_class(args.ngames))
    final_reward = my_runner.loop(args.ngames, args.niter)
    print ("Obtained a final reward of {}".format(final_reward))



if __name__ == "__main__":
    main()
