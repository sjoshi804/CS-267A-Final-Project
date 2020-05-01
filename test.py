import gym
from gym.wrappers import Monitor
import gym_pursuit_evasion
import sys
import torch
import pyro
from time import sleep as sleep

def main(argv=()):
    del argv  # Unused.

    # Build an environment
    
    # Record episode
    env = Monitor(gym.make('one-stationary-evader-v0'), './tmp/pursuit_evasion_semi_random_pursuer_vs_stationary_evader', force=True)

    # Don't record episode 
    # env = gym.make('one-random-evader-v0')

    state = env.reset()

    while True:

        #Render
        env.render()

        #Agent goes here
        action_distribution = torch.distributions.Categorical(torch.tensor([0, 0, 0.5, 0.5]))
        
        action = action_distribution.sample().item()

        #Delay to make video easier to watch
        #sleep(5)

        #Get observations, rewards, termination form environment after taking action
        observation, reward, done, info = env.step(action) 

        if done: 
            break

    env.close()

if __name__ == '__main__':
  main(sys.argv)
