# CS 267A Project - Reinforcement Learning using PyColab Game Engine

## Setup

```
pip3 install -r requirements.txt
```

## Running Pursuit Evasion with Random Evader and Human Pursuer

This runs a game set in the 4 rooms grid with a human pursuer controlled by 
left right up down keys who is chasing a random evader. The reward for catching
the evader is 100 and cost of every second is -1. 

```
python3 pursuit_evasion.py
```

## Running PyColab Examples

```
python3 examples/four_rooms.py

python3 examples/cliff_walk.py

python3 examples/chain_walk.py

python3 examples/apprehend.py
```