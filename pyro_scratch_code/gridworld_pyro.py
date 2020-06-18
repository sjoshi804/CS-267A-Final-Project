"""

 This code is based on the code at: https://github.com/ankonzoid/LearningX/blob/master/classical_RL/gridworld/gridworld.py 

"""

import os, sys, random, operator
import numpy as np
import gym
from gym.wrappers import Monitor
import gym_pursuit_evasion
import sys
import torch
import torch.distributions as dist
from time import sleep as sleep
import dice_inference_engine as infer

class Environment:
    
    def __init__(self, Ny=2, Nx=9):
        # Define state space
        self.Ny = Ny  # y grid size
        self.Nx = Nx  # x grid size
        self.state_dim = (Ny, Nx)
        # Define action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "down": 1, "left": 2, "right": 3}
        self.action_coords = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # translations
        # Define rewards table
        self.R = self._build_rewards()  # R(s,a) agent rewards
        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")

    def reset(self):
        # Reset agent state to top-left grid corner
        self.state = (0, 0)  
        return self.state

    def step(self, action):
        # Evolve agent state
        state_next = (self.state[0] + self.action_coords[action][0],
                      self.state[1] + self.action_coords[action][1])
        # Collect reward
        reward = self.R[self.state + (action,)]
        # Terminate if we reach bottom-right grid corner
        done = (state_next[0] == self.Ny - 1) and (state_next[1] == self.Nx - 1)
        # Update state
        self.state = state_next
        return state_next, reward, done
    
    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = self.state[0], self.state[1]
        if (y > 0):  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (y < self.Ny - 1):  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (x > 0):  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])
        if (x < self.Nx - 1):  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed

    def _build_rewards(self):
        # Define agent rewards R[s,a]
        r_goal = 100  # reward for arriving at terminal state (bottom-right corner)
        r_nongoal = -0.1  # penalty for not reaching terminal state
        R = r_nongoal * np.ones(self.state_dim + self.action_dim, dtype=float)  # R[s,a]
        R[self.Ny - 2, self.Nx - 1, self.action_dict["down"]] = r_goal  # arrive from above
        R[self.Ny - 1, self.Nx - 2, self.action_dict["right"]] = r_goal  # arrive from the left
    
        return R
        
class Environment_gym:
    
    def __init__(self):
        # Build an environment
        
        # Create and record episode - remove Monitor statement if recording not desired
        self.env = Monitor(gym.make('one-stationary-evader-v0'), './tmp/pursuit_evasion_infer_pursuer_vs_stationary_evader', force=True)
    
        

    def reset(self):
        # Reset agent state to top-left grid corner
        #Reset state
        self.state = self.env.reset()
        
        #Initialize Agent Parameters
        #Get observed state space
        self.observed_state_space = self.env.get_observed_state_space()
        #Set initial state distribution
        self.initial_state_dist = []
        self.initial_state = self.env.get_initial_state()
        for self.state in self.observed_state_space:
            if self.state == self.initial_state:
                self.initial_state_dist.append(1)
            else:
                self.initial_state_dist.append(0) 
        return self.state


class Agent:
    
    def __init__(self, env):
        # Store state and action dimension 
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        # Agent learning parameters
        self.epsilon = 1  # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.beta = 0.99  # learning rate
        self.gamma = 0.99  # reward discount factor
        # Initialize Q[s,a] table
        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)

    def get_action(self, env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:
            # exploit on allowed actions
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):
        # -----------------------------
        # Update:
        #
        # Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])
        #
        #  R[s,a] = reward for taking action a from state s
        #  beta = learning rate
        #  gamma = discount factor
        # -----------------------------
        (state, action, state_next, reward, done) = memory
        sa = state + (action,)
        self.Q[sa] += self.beta * (reward + self.gamma*np.max(self.Q[state_next]) - self.Q[sa])

    def display_greedy_policy(self):
        # greedy policy = argmax[a'] Q[s,a']
        greedy_policy = np.zeros((self.state_dim[0], self.state_dim[1]), dtype=int)
        for x in range(self.state_dim[0]):
            for y in range(self.state_dim[1]):
                greedy_policy[x, y] = np.argmax(self.Q[x, y, :])
        print("\nGreedy policy(y, x):")
        print(greedy_policy)
        print()

# Settings
env_pyro = Environment(Ny=2, Nx=9)
agent = Agent(env_pyro)

# Build an environment
    
# Create and record episode - remove Monitor statement if recording not desired
env = Monitor(gym.make('one-stationary-evader-v0'), './tmp/pursuit_evasion_infer_pursuer_vs_stationary_evader', force=True)

#Reset state
state_gym = env.reset()

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
max_trajectory_length = 11 #needs to be greater than shortest distance to evader for any meaningful inference


# Train agent
print("\nTraining agent...\n")
N_episodes = 600
for episode in range(N_episodes):

    # Generate an episode
    iter_episode, reward_episode = 0, 0
    state = env_pyro.reset()  # starting state
    action = 0
    while True:
      action = agent.get_action(env_pyro)  # get action
      state_next, reward, done = env_pyro.step(action)  # evolve state by action
      agent.train((state, action, state_next, reward, done))  # train agent
      iter_episode += 1
      reward_episode += reward
      if (episode == N_episodes-1):
        env.render()
        #sleep(1)
        print(action)
        observation1, reward1, done, info1 = env.step(action)
        sleep(1)
      
      if done:
        break
      state = state_next  # transition to next state

    # Decay agent exploration parameter
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

    # Print
    if (episode == 0) or (episode + 1) % 10 == 0:
        print("[episode {}/{}] eps = {:.3F} -> iter = {}, rew = {:.1F}".format(
            episode + 1, N_episodes, agent.epsilon, iter_episode, reward_episode))

    #Print greedy policy
    if (episode == N_episodes - 1):
        agent.display_greedy_policy()
        for (key, val) in sorted(env_pyro.action_dict.items(), key=operator.itemgetter(1)):
            print(" action['{}'] = {}".format(key, val))
        print()