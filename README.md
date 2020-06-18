# CS 267A Project - Reinforcement Learning using PyColab Game Engine

## Setup

```
pip3 install -e gym-pursuit-evasion
```

## Running the Inference Based Agent using Dice

This is only supported for the stationary evader environment as of now. 
Running the command below, has an inference based agent catch an evader
in a simple grid based environment. The agent is incentivised to do this
quickly due to the reward for each time step when the goal is not reached 
to be -10 and the reward for reaching the goal = 0. 
Note for the inference based agent, rewards must be negative due to the
paradigm we are using which is explained here: https://arxiv.org/pdf/1805.00909.pdf

```
python3 -W ignore agent_dice.py
```

## Running The Inference using Pyro

This is only supported for the stationary evader environment as of now. 

```
python3 -W ignore agent_pyro.py
```

## Running Pursuit Evasion with Random Evader and Human Pursuer

This runs a game set in the 4 rooms grid with a human pursuer controlled by 
left right up down keys who is chasing a random evader. The reward for catching
the evader is 100 and cost of every second is -1. 

```
python3 -W ignore examples/pursuit_evasion.py
```

## Running PyColab Examples

```
python3 -W ignore examples/four_rooms.py

python3 -W ignore examples/cliff_walk.py

python3 -W ignore examples/chain_walk.py

python3 -W ignore examples/apprehend.py
```
