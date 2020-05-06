import gym
from gym.wrappers import Monitor
import gym_pursuit_evasion
import sys
import torch.distributions as dist
from time import sleep as sleep
import dice_inference_engine as infer

def main(argv=()):
    del argv  # Unused.

    # Build an environment
    
    # Create and record episode - remove Monitor statement if recording not desired
    env = Monitor(gym.make('one-stationary-evader-v0'), './tmp/pursuit_evasion_semi_random_pursuer_vs_stationary_evader', force=True)

    #Reset state
    state = env.reset()
    
    #Initialize Agent Parameters
    #Get observed state space
    observed_state_space = env.get_observed_state_space()
    #Set initial state distribution
    initial_state_dist = []
    initial_state = env.get_initial_state()
    for state in observed_state_space:
        if state == initial_state:
            initial_state_dist.append(1)
        else:
            initial_state_dist.append(0)
    #Get action space
    action_space = range(0, env.action_space.n)
    #Set action prior to uniform dist
    action_prior = []
    for action in action_space:
        action_prior.append(1/len(action_space))
    #Get reward function
    reward_function = env.get_reward_function()
    #Get transition function 
    transition_function = env.get_transition_function()
    #Set max trajectory length
    max_trajectory_length = 11

    #Create Agent
    agent = infer.DiceInferenceEngine(observed_state_space, action_space, initial_state_dist, action_prior, reward_function, transition_function, max_trajectory_length)

    #Set current observed state to initial state
    observed_state = initial_state

    #Game Loop
    for t in range(0, 11):

        #Render
        env.render()

        #Pick action based on agent's reasoning
        action_dist = agent.next(observed_state)
        print(action_dist)
        #Delay to make video easier to watch
        #sleep(5)

        #Get observations, rewards, termination form environment after taking action
       #  _, reward, done, info = env.step(action) 
 

    env.close()

if __name__ == '__main__':
  main(sys.argv)
