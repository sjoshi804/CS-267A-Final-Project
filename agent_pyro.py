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

pyro.set_rng_seed(101)

state = (0, 0)
final_state = (1,2)
len_trajectory = 3
HEIGHT = 2
WIDTH = 9

action_dict = {"up": 0, "down": 1, "left": 2, "right": 3}
action_coords = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # translations
        
        
def reward_function(state):
    if state is not None and state == final_state:
        reward = 0.0
    else:
        reward = -10.0
    return reward
    
def transition(state, action):
    next_state = (state[0] + action_coords[action][0],
                      state[1] + action_coords[action][1])
    return next_state


# for a grid of HEIGHT * WIDTH
def unhash_state(state):
    return (int(state // WIDTH), int(state % WIDTH))


def convert_to_prob(state):
    num = HEIGHT * WIDTH
    probs = [0.0] * num
    for i in range(0, num):
        if unhash_state(i) == state:
            probs[i] = float(1)
            break   
    ## if the state is not among the valid states, assign prob of 1 to a random state
    if (sum(probs) != float(1)):
        ind = random.randint(0, 18)
        probs[ind-1] = float(1)
    return probs

def agent_model(state=(0, 0), len_trajectory=1):  
    state_1 = state
    for i in range(len_trajectory):
      p = torch.ones(4)/4
      action = pyro.sample("action_{}".format(i), dist.Categorical(p)) # up, down, left, right
      prob_1 = convert_to_prob(transition(state_1, action))    
      state_1 = unhash_state(pyro.sample("state_{}".format(i), dist.Categorical(torch.tensor(prob_1))))
      pyro.sample("optimal_{}".format(i), dist.Bernoulli(torch.tensor([exp(reward_function(state_1))])), obs=torch.tensor([1.]))



def agent_guide(state, len_trajectory):
    state_1 = state
    for i in range(len_trajectory):
      p_guide = pyro.param("p_guide_{}".format(i), torch.ones(4)/4, constraint=constraints.simplex)
      action = pyro.sample("action_{}".format(i), dist.Categorical(p_guide))
      prob_1 = convert_to_prob(transition(state_1, action))
      state_1 = unhash_state(pyro.sample("state_{}".format(i), dist.Categorical(torch.tensor(prob_1))))


pyro.clear_param_store()
svi = pyro.infer.SVI(model=agent_model,
                     guide=agent_guide,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.01}),
                     loss=pyro.infer.Trace_ELBO())

num_steps = 5000
losses = []
for t in range(num_steps):
    losses.append(svi.step(state, len_trajectory))

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
plt.show()

for i in range(len_trajectory):
  print('p_guide_{} = '.format(i), pyro.param("p_guide_{}".format(i)))