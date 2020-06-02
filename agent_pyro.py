import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import constraints
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from math import exp as exp

pyro.set_rng_seed(101)

state = torch.tensor(0.)
next_state = torch.tensor(1.)
def reward_function(state):
    if state is not None and torch.equal(state, torch.tensor(3.)):
        reward = 0.0
    else:
        reward = -10.0
    return reward
    
def transition(action, state, next_state):
    next_state = state.add(action)
      
      
def encoder_action(state, next_state):
    t = next_state.sub(state)
    if torch.equal(t, torch.tensor(0.)):
      return torch.tensor([1.0, 0.0, 0.0, 0.0])
    elif torch.equal(t, torch.tensor(1.)):
      return torch.tensor([0.0, 1.0, 0.0, 0.0])
    elif torch.equal(t, torch.tensor(2.)):
      return torch.tensor([0.0, 0.0, 1.0, 0.0])
    else :
      return torch.tensor([0.0, 0.0, 0.0, 1.0])
      
    
      
      
def agent_model(state=None, next_state=None):  
    #pyro.module("transition", transition)
    
    p = pyro.param("p", torch.ones(4)/4, constraint=constraints.simplex)
    action = pyro.sample("action", dist.Categorical(p)) # up, down, left, right
    #state = pyro.param("state", torch.tensor([0, 0]))
    #next_state = pyro.param("state", torch.tensor([0, 0]))
    state = pyro.param("state", torch.tensor(0.))
    next_state = torch.tensor(2.)
    return pyro.sample("optimal", dist.Bernoulli(torch.tensor([exp(reward_function(transition(action, state, next_state)))])), obs=torch.tensor([1.]))


def agent_guide(state, next_state):
    action = pyro.sample("action", dist.Categorical(encoder_action(state, next_state)))
    return action

pyro.clear_param_store()
svi = pyro.infer.SVI(model=agent_model,
                     guide=agent_guide,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                     loss=pyro.infer.Trace_ELBO())


num_steps = 1000
print(agent_model(state, next_state))
losses = []
for t in range(num_steps):
    losses.append(svi.step(state, next_state))
    #print(state)

#plt.plot(losses)
#plt.title("ELBO")
#plt.xlabel("step")
#plt.ylabel("loss")
#plt.show()
#print('Action = ',pyro.param("action").item())
print('State = ',pyro.param("state").item())
#print('Next State = ',pyro.param("next_state").item())
print('p = ', pyro.param("p"))
