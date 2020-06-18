# CS 267A Project - Reinforcement Learning as Inference using Probabilistic Programming 

## Setup

```
pip3 install -e gym-pursuit-evasion
```

## Running the Inference Based Agent using Dice

By default this runs in the random evader environment. 
Running the command below, has an inference based agent catch an evader
in a simple grid based environment. The agent is incentivised to do this
quickly due to the reward for each time step when the goal is not reached 
to be -10 and the reward for reaching the goal = 0. 
Note for the inference based agent, rewards must be negative due to the
paradigm we are using which is explained here: https://arxiv.org/pdf/1805.00909.pdf

```
python3 -W ignore agent-dice.py
```

## Running the Inference Based Agent using Pyro

By default this runs in the stationary evader environment. 
This command executes an agent that is modeled identical to the previous command but is written in Pyro

```
python3 -W ignore agent-dice.py
```

## Py2Dice

The py2dice directory contains the code for dynamically creating ASTs in Dice using Python and using them to progammatically generate Dice programs using Python. The Dice Program Generator in dice_inference_engine.py however for simplicity doesn't rely on this, but this tool is more generally applicable and will hopefully serve as a way to extend the functionality of Dice. 

## Miscellaneous

- The tmp directory is where the videos generated from executing the agent commands are stored. This can be useful to observe how the agent is doing. 
- The .infer directory is where the dynamically created Dice programs are stored. 
- The gym-pursuit-evasion directory contains the Open AI Gym Wrapped Environment created for Reinforcement Learning in this project. 
- The dice_scratch_code directory contains the scratch code written in designing the Dice Inference Engine
- The pyro_scratch_code directory contains the scratch code written in designing the Pyro-based agent