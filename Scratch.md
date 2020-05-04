# Scratch

## Dice-Powered RL as Inference


### Rough sketch
Dynamically create Dice Programs to represent appropriate probability distirbutions
Parameters: reward function, state transistion dynamics of the environment, action prior (assume to be uniform?), trajectory max length

### Possible Challenges

- Terminating seqeuence if goal state achieved - no need to do any further inference - do we need to do this manually or does dice implicitly do this? 

- Interfacing with python program live during an execution or compiling the probability distribution and storing in a python program? 

