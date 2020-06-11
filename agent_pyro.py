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

def unhash_state2(state):
    if state == 0:
        return (-2, 0)
    elif state == 1:
        return (-1, -1)
    elif state == 2:
        return (-1, 0)
    elif state == 3:
        return (-1, 1)
    elif state == 4:
        return (0, -2)
    elif state == 5:
        return (0, -1)
    elif state == 6:
        return (0, 0)
    elif state == 7:
        return (0, 1)
    elif state == 8:
        return (0, 2)
    elif state == 9:
        return (1, -1)
    elif state == 10:
        return (1, 0)
    elif state == 11:
        return (1, 1)
    elif state == 12:
        return (2, 0)

def unhash_state(state):
    #print(state)
    return (int(state / 9), int(state % 9))


def convert_to_prob(state):
    probs = [0.0] * 18
    for i in range(0, 18):
        if unhash_state(i) == state:
            probs[i] = float(1)
            break   
    if (sum(probs) != float(1)):
        ind = random.randint(0, 18)
        probs[ind-1] = float(1)
    return probs

def agent_model(state=(0, 0)):  
    p = torch.ones(4)/4
    action = pyro.sample("action", dist.Categorical(p)) # up, down, left, right
    prob_1 = convert_to_prob(transition(state, action))    
    state_1 = unhash_state(pyro.sample("state_1", dist.Categorical(torch.tensor(prob_1))))
    pyro.sample("optimal", dist.Bernoulli(torch.tensor([exp(reward_function(state_1))])), obs=torch.tensor([1.]))
    
    p2 = torch.ones(4)/4
    action2 = pyro.sample("action2", dist.Categorical(p2))
    prob_2 = convert_to_prob(transition(state_1, action2))
    state_2 = unhash_state(pyro.sample("state_2", dist.Categorical(torch.tensor(prob_2))))
    pyro.sample("optimal2", dist.Bernoulli(torch.tensor([exp(reward_function(state_2))])), obs=torch.tensor([1.]))
    
    p3 = torch.ones(4)/4
    action3 = pyro.sample("action3", dist.Categorical(p3))
    prob_3 = convert_to_prob(transition(state_2, action3))
    state_3 = unhash_state(pyro.sample("state_3", dist.Categorical(torch.tensor(prob_3))))
    return pyro.sample("optimal3", dist.Bernoulli(torch.tensor([exp(reward_function(state_3))])), obs=torch.tensor([1.]))


def agent_guide(state):
    #action = pyro.sample("action", dist.Categorical(encoder_action(state, next_state)))
    p_guide = pyro.param("p_guide", torch.ones(4)/4, constraint=constraints.simplex)
    action = pyro.sample("action", dist.Categorical(p_guide))
    prob_1 = convert_to_prob(transition(state, action))
    state_1 = unhash_state(pyro.sample("state_1", dist.Categorical(torch.tensor(prob_1))))
    
    p2_guide = pyro.param("p2_guide", torch.ones(4)/4, constraint=constraints.simplex)
    action2 = pyro.sample("action2", dist.Categorical(p2_guide))
    prob_2 = convert_to_prob(transition(state_1, action2))
    state_2 = unhash_state(pyro.sample("state_2", dist.Categorical(torch.tensor(prob_2))))
    
    p3_guide = pyro.param("p3_guide", torch.ones(4)/4, constraint=constraints.simplex)
    action3 = pyro.sample("action3", dist.Categorical(p3_guide))
    prob_3 = convert_to_prob(transition(state_2, action3))
    state_3 = unhash_state(pyro.sample("state_3", dist.Categorical(torch.tensor(prob_3))))
    #return action

#print(unhash_state(16))
pyro.clear_param_store()
svi = pyro.infer.SVI(model=agent_model,
                     guide=agent_guide,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                     loss=pyro.infer.Trace_ELBO())

num_steps = 5000
#print(agent_model(state, next_state))
losses = []
for t in range(num_steps):
    losses.append(svi.step(state))
    #print(state)

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
plt.show()

print('p_guide = ', pyro.param("p_guide"))
print('p2_guide = ', pyro.param("p2_guide"))
print('p3_guide = ', pyro.param("p2_guide"))