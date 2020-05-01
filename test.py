import gym
from gym.wrappers import Monitor
import gym_pursuit_evasion
import sys

def main(argv=()):
    del argv  # Unused.

    # Build a four-rooms game.
    
    # Record episode
    env = Monitor(gym.make('one-random-evader-v0'), './tmp/pursuit_evasion_random_pursuer_vs_random_evader', force=True)

    # Don't record episode 
    # env = gym.make('one-random-evader-v0')

    state = env.reset()

    while True:

        #Render
        env.render()

        #Agent goes here
        action = env.action_space.sample()

        #Get observations, rewards, termination form environment after taking action
        observation, reward, done, info = env.step(action) 

        #Delay to make video easier to watch
        #sleep(0.5)

        if done: 
            break

    env.close()

if __name__ == '__main__':
  main(sys.argv)
