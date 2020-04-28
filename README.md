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
python3 -W ignore pursuit_evasion.py
```

## Running Pursuit Evasion with Random Evader and Random Pursuer

This runs a game set in the 4 rooms grid with a random pursuer controlled by 
left right up down keys who is chasing a random evader. The reward for catching
the evader is 100 and cost of every second is -1. This game is wrapped with a gym
environment agents to be extended to be more sophisticated RL agents easily. The 
episode can be observed as a video as it occurs and is recorded for later viewing 
in the tmp folder. 

```
python3 -W ignore gym_pursuit_evasion.py
```

## Running PyColab Examples

```
python3 -W ignore examples/four_rooms.py

python3 -W ignore examples/cliff_walk.py

python3 -W ignore examples/chain_walk.py

python3 -W ignore examples/apprehend.py
```