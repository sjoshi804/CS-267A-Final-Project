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

def reward_function(action):
    if action == torch.tensor([3]):
        reward = 0
    else:
        reward = -2
    return reward

def agent_model():  
    action = pyro.sample("action", dist.Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25])))
    return pyro.sample("optimal", dist.Bernoulli(torch.tensor([exp(reward_function(action))])), obs=1)

def agent_guide():
    up = pyro.param("up", torch.tensor(0.25))
    down = pyro.param("down", torch.tensor(0.25))
    left = pyro.param("left", torch.tensor(0.25))
    right = pyro.param("right", torch.tensor(1 - up - down - left))
    action = pyro.sample("action", dist.Categorical(torch.tensor([up, down, left, right])))
    return pyro.sample("optimal", dist.Bernoulli(torch.tensor([exp(reward_function(action))])))

pyro.clear_param_store()
svi = pyro.infer.SVI(model=agent_model,
                     guide=agent_guide,
                     optim=pyro.optim.SGD({"lr": 0.1, "momentum":0.0001}),
                     loss=pyro.infer.Trace_ELBO())


num_steps = 1000
losses = []
for t in range(num_steps):
    losses.append(svi.step())

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
plt.show()
print('UP = ',pyro.param("up").item())
print('DOWN = ',pyro.param("down").item())
print('LEFT = ',pyro.param("left").item())
print('RIGHT = ',pyro.param("right").item())
