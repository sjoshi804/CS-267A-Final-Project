import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import constraints
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from math import exp as exp
import random
import os, sys, random, operator
import gym
from gym.wrappers import Monitor
import gym_pursuit_evasion
from time import sleep as sleep

pyro.set_rng_seed(101)

state = (0, 0)
final_state = (1,8)
len_trajectory = 5
HEIGHT = 2
WIDTH = 9

action_dict = {"up": 0, "down": 1, "left": 2, "right": 3}
action_coords = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # translations
        
        
def reward_function(state):
    if state is not None and state == final_state:
        reward = 0.0
    else:
        reward = -100.0
    return reward
    
def allowed_actions(state, Ny = HEIGHT, Nx=WIDTH):
    # Generate list of actions allowed depending on agent grid location
    actions_allowed = []
    y, x = state[0], state[1]
    if (y > 0):  # no passing top-boundary
        actions_allowed.append(action_dict["up"])
    if (y < Ny - 1):  # no passing bottom-boundary
        actions_allowed.append(action_dict["down"])
    if (x > 0):  # no passing left-boundary
        actions_allowed.append(action_dict["left"])
    if (x < Nx - 1):  # no passing right-boundary
        actions_allowed.append(action_dict["right"])
    actions_allowed = np.array(actions_allowed, dtype=int)
    return actions_allowed
    
def transition(state, action):
    next_state = (state[0] + action_coords[action][0],
                      state[1] + action_coords[action][1])
    return next_state


# for a grid of HEIGHT * WIDTH
def unhash_state(state):
    return (int(state // WIDTH), int(state % WIDTH))


def convert_to_prob(state, prev_state):
    num = HEIGHT * WIDTH
    probs = [0.0] * num
    for i in range(0, num):
        if unhash_state(i) == state:
            probs[i] = float(1)
            break   
    ## if the state is not among the valid states, assign prob of 1 to a random adjacent state
    if (sum(probs) != float(1)):
        direction = random.choice(allowed_actions(prev_state))
        probs = convert_to_prob(transition(prev_state, direction), prev_state)

    return probs

def agent_model(state=(0, 0), len_trajectory=1):  
    state_1 = state
    for i in range(len_trajectory):
      p = torch.ones(4)/4
      action = pyro.sample("action_{}".format(i), dist.Categorical(p)) # up, down, left, right
      prob_1 = convert_to_prob(transition(state_1, action), state_1)    
      state_1 = unhash_state(pyro.sample("state_{}".format(i), dist.Categorical(torch.tensor(prob_1))))
      pyro.sample("optimal_{}".format(i), dist.Bernoulli(torch.tensor([exp(reward_function(state_1))])), obs=torch.tensor([1.]))



def agent_guide(state, len_trajectory):
    state_1 = state
    for i in range(len_trajectory):
      p_guide = pyro.param("p_guide_{}".format(i), torch.ones(4)/4, constraint=constraints.simplex)
      action = pyro.sample("action_{}".format(i), dist.Categorical(p_guide))
      prob_1 = convert_to_prob(transition(state_1, action), state_1)
      state_1 = unhash_state(pyro.sample("state_{}".format(i), dist.Categorical(torch.tensor(prob_1))))



      
# Build an environment
    
# Create and record episode - remove Monitor statement if recording not desired
env = Monitor(gym.make('one-stationary-evader-v0'), './tmp/pursuit_evasion_infer_pursuer_vs_stationary_evader', force=True)

##Reset state
state_gym = env.reset()

current_state = state
step_ = 0
while (current_state != final_state):
  print("############################")
  print("Inferring new set of actions")
  print("############################")
  print()
  pyro.clear_param_store()
  svi = pyro.infer.SVI(model=agent_model,
                       guide=agent_guide,
                       optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.01}),
                       loss=pyro.infer.Trace_ELBO())
  
  num_steps = 6000
  losses = []
  for t in range(num_steps):
      losses.append(svi.step(current_state, len_trajectory))
      
  for i in range(len_trajectory):
    #print('p_guide_{} = '.format(i), torch.max(pyro.param("p_guide_{}".format(i)), 0))
    _, action = torch.max(pyro.param("p_guide_{}".format(i)), 0)
    current_state = transition(current_state, action)
    env.render()
    observation1, reward1, done, info1 = env.step(action)
    env.render()
    sleep(1)
    step_ += 1
    if done:
      print("******************")
      print("Reached the evader")
      print("******************")
      current_state = final_state
      print()
      break
    if current_state == final_state:
      break
      
print(step_)


#plt.plot(losses)
#plt.title("ELBO")
#plt.xlabel("step")
#plt.ylabel("loss")
#plt.show()

