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

state = (0, 0)
final_state = (1,-1)

action_dict = {"up": 0, "down": 1, "left": 2, "right": 3}
action_coords = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # translations
        
        
def reward_function(state):
    if state is not None and state == final_state:
        reward = 0.0
    else:
        reward = -10.0
    return reward
    
def transition(action, state):
    #next_state = state.add(action)
    next_state = (state[0] + action_coords[action][0],
                      state[1] + action_coords[action][1])
    return next_state
      
      
def encoder_action(state, next_state):
    t = (next_state[0] - state[0], 
         next_state[1] - state[1])
    if t == (-1, 0):
      return torch.tensor([1.0, 0.0, 0.0, 0.0])
    elif t == (1, 0):
      return torch.tensor([0.0, 1.0, 0.0, 0.0])
    elif t == (0, -1):
      return torch.tensor([0.0, 0.0, 1.0, 0.0])
    else :
      return torch.tensor([0.0, 0.0, 0.0, 1.0])
#    t = next_state.sub(state)
#    if torch.equal(t, torch.tensor(0.)):
#      return torch.tensor([1.0, 0.0, 0.0, 0.0])
#    elif torch.equal(t, torch.tensor(1.)):
#      return torch.tensor([0.0, 1.0, 0.0, 0.0])
#    elif torch.equal(t, torch.tensor(2.)):
#      return torch.tensor([0.0, 0.0, 1.0, 0.0])
#    else :
#      return torch.tensor([0.0, 0.0, 0.0, 1.0])
      
    
      
      
def agent_model(state=None, final_state=None):  
    #pyro.module("transition", transition)
    
    p = pyro.param("p", torch.ones(4)/4, constraint=constraints.simplex)
    action = pyro.sample("action", dist.Categorical(p)) # up, down, left, right
    pyro.sample("optimal", dist.Bernoulli(torch.tensor([exp(reward_function(transition(action, state)))])), obs=torch.tensor([1.]))
    
    p2 = pyro.param("p2", torch.ones(4)/4, constraint=constraints.simplex)
    action2 = pyro.sample("action2", dist.Categorical(p2)) # up, down, left, right
    return pyro.sample("optimal2", dist.Bernoulli(torch.tensor([exp(reward_function(transition(action2, transition(action, state))))])), obs=torch.tensor([1.]))


def agent_guide(state, next_state):
    #action = pyro.sample("action", dist.Categorical(encoder_action(state, next_state)))
    p_guide = pyro.param("p_guide", torch.ones(4)/4, constraint=constraints.simplex)
    action = pyro.sample("action", dist.Categorical(p_guide))
    
    p2_guide = pyro.param("p2_guide", torch.ones(4)/4, constraint=constraints.simplex)
    action2 = pyro.sample("action2", dist.Categorical(p2_guide))
    #return action

pyro.clear_param_store()
svi = pyro.infer.SVI(model=agent_model,
                     guide=agent_guide,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                     loss=pyro.infer.Trace_ELBO())


num_steps = 3000
#print(agent_model(state, next_state))
losses = []
for t in range(num_steps):
    losses.append(svi.step(state, final_state))
    #print(state)

#plt.plot(losses)
#plt.title("ELBO")
#plt.xlabel("step")
#plt.ylabel("loss")
#plt.show()
#print('Action = ',pyro.param("action").item())
#print('State = ',pyro.param("state").item())
#print('Next State = ',pyro.param("next_state").item())
print('p = ', pyro.param("p"))
print('p_guide = ', pyro.param("p_guide"))
print('p2 = ', pyro.param("p2"))
print('p2_guide = ', pyro.param("p2_guide"))
