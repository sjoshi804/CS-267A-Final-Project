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
    
def transition(state, action):
    #next_state = state.add(action)
    next_state = (state[0] + action_coords[action][0],
                      state[1] + action_coords[action][1])
    return next_state

def unhash_state(state):
    if state == 0:
        return (2, 0)
    elif state == 1:
        return (1, 1)
    elif state == 2:
        return (0, 2)
    elif state == 3:
        return (1, 0)
    elif state == 4:
        return (0, 1)
    elif state == 5:
        return (0, 0)
    elif state == 6:
        return (-1, 0)
    elif state == 7:
        return (0, -1)
    elif state == 8:
        return (-1, -1)
    elif state == 9:
        return (-2, 0)
    elif state == 10:
        return (0, -2)
    elif state == 11:
        return (1, -1)
    elif state == 12:
        return (-1, 1)

def convert_to_prob(state):
    probs = []
    for i in range(0, 13):
        if unhash_state(i) == state:
            probs.append(float(1))
        else:
            probs.append(float(0))
    return probs

def agent_model(state=(0, 0)):  
    p = torch.ones(4)/4
    action = pyro.sample("action", dist.Categorical(p)) # up, down, left, right
    state_1 = unhash_state(pyro.sample("state_1", dist.Categorical(torch.tensor(convert_to_prob(transition(state, action))))))
    pyro.sample("optimal", dist.Bernoulli(torch.tensor([exp(reward_function(state_1))])), obs=torch.tensor([1.]))
    
    p2 = torch.ones(4)/4
    action2 = pyro.sample("action2", dist.Categorical(p2))
    state_2 = unhash_state(pyro.sample("state_2", dist.Categorical(torch.tensor(convert_to_prob(transition(state_1, action2))))))
    return pyro.sample("optimal2", dist.Bernoulli(torch.tensor([exp(reward_function(state_2))])), obs=torch.tensor([1.]))


def agent_guide(state):
    #action = pyro.sample("action", dist.Categorical(encoder_action(state, next_state)))
    p_guide = pyro.param("p_guide", torch.ones(4)/4, constraint=constraints.simplex)
    action = pyro.sample("action", dist.Categorical(p_guide))
    state_1 = unhash_state(pyro.sample("state_1", dist.Categorical(torch.tensor(convert_to_prob(transition(state, action))))))
    p2_guide = pyro.param("p2_guide", torch.ones(4)/4, constraint=constraints.simplex)
    action2 = pyro.sample("action2", dist.Categorical(p2_guide))
    state_2 = unhash_state(pyro.sample("state_2", dist.Categorical(torch.tensor(convert_to_prob(transition(state_1, action2))))))
    #return action

pyro.clear_param_store()
svi = pyro.infer.SVI(model=agent_model,
                     guide=agent_guide,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                     loss=pyro.infer.Trace_ELBO())

num_steps = 10000
#print(agent_model(state, next_state))
losses = []
for t in range(num_steps):
    losses.append(svi.step(state))
    #print(state)

#plt.plot(losses)
#plt.title("ELBO")
#plt.xlabel("step")
#plt.ylabel("loss")
#plt.show()

print('p_guide = ', pyro.param("p_guide"))
print('p2_guide = ', pyro.param("p2_guide"))
