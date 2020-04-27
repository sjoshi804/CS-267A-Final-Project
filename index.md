---
layout: default
title: CS 267A Project
description: Final Project for UCLA 267A - Probabilistic Progamming and Relational Learning
is_section: true
---

## Introduction

Examples and environment borrowed from www.agentmodels.org

## Example 1

<!-- show_single_step_trajectory -->
~~~~
///fold: restaurant constants
var ___ = ' ';
var DN = { name: 'Donut N' };
var DS = { name: 'Donut S' };
var V = { name: 'Veg' };
var N = { name: 'Noodle' };
///

var grid = [
  ['#', '#', '#', '#',  V , '#'],
  ['#', '#', '#', ___, ___, ___],
  ['#', '#', DN , ___, '#', ___],
  ['#', '#', '#', ___, '#', ___],
  ['#', '#', '#', ___, ___, ___],
  ['#', '#', '#', ___, '#',  N ],
  [___, ___, ___, ___, '#', '#'],
  [DS , '#', '#', ___, '#', '#']
];

var world = makeGridWorldMDP({ grid }).world;

var trajectory = [
  {
    loc: [3, 1],
    timeLeft: 11,
    terminateAfterAction: false
  },
  {
    loc: [2, 1],
    timeLeft: 10,
    terminateAfterAction: false
  }
];

viz.gridworld(world, { trajectory });
~~~~

## Example 2

<!-- infer_from_single_step_trajectory -->
~~~~
///fold: create restaurant choice MDP
var ___ = ' ';
var DN = { name : 'Donut N' };
var DS = { name : 'Donut S' };
var V = { name : 'Veg' };
var N = { name : 'Noodle' };

var grid = [
  ['#', '#', '#', '#',  V , '#'],
  ['#', '#', '#', ___, ___, ___],
  ['#', '#', DN , ___, '#', ___],
  ['#', '#', '#', ___, '#', ___],
  ['#', '#', '#', ___, ___, ___],
  ['#', '#', '#', ___, '#',  N ],
  [___, ___, ___, ___, '#', '#'],
  [DS , '#', '#', ___, '#', '#']
];

var mdp = makeGridWorldMDP({
  grid,
  noReverse: true,
  maxTimeAtRestaurant: 2,
  start: [3, 1],
  totalTime: 11
});
///

var world = mdp.world;
var makeUtilityFunction = mdp.makeUtilityFunction;

var utilityTablePrior = function(){
  var baseUtilityTable = {
    'Donut S': 1,
    'Donut N': 1,
    'Veg': 1,
    'Noodle': 1,
    'timeCost': -0.04
  };
  return uniformDraw(
    [{ table: extend(baseUtilityTable, { 'Donut N': 2, 'Donut S': 2 }),
       favourite: 'donut' },
     { table: extend(baseUtilityTable, { 'Veg': 2 }),
       favourite: 'veg' },
     { table: extend(baseUtilityTable, { 'Noodle': 2 }),
       favourite: 'noodle' }]
  );
};

var observedTrajectory = [[{
  loc: [3, 1],
  timeLeft: 11,
  terminateAfterAction: false
}, 'l']];

var posterior = Infer({ model() {
  var utilityTableAndFavourite = utilityTablePrior();
  var utilityTable = utilityTableAndFavourite.table;
  var utility = makeUtilityFunction(utilityTable);
  var favourite = utilityTableAndFavourite.favourite;

  var agent  = makeMDPAgent({ utility, alpha: 2 }, world);
  var act = agent.act;

  // For each observed state-action pair, factor on likelihood of action
  map(
    function(stateAction){
      var state = stateAction[0];
      var action = stateAction[1];
      observe(act(state), action);
    },
    observedTrajectory);

  return { favourite };
}});

viz(posterior);
~~~~

## Example 3

~~~~
// infer_utilities_timeCost_softmax_noise
///fold: create restaurant choice MDP, donutSouthTrajectory
var ___ = ' ';
var DN = { name : 'Donut N' };
var DS = { name : 'Donut S' };
var V = { name : 'Veg' };
var N = { name : 'Noodle' };

var grid = [
  ['#', '#', '#', '#',  V , '#'],
  ['#', '#', '#', ___, ___, ___],
  ['#', '#', DN , ___, '#', ___],
  ['#', '#', '#', ___, '#', ___],
  ['#', '#', '#', ___, ___, ___],
  ['#', '#', '#', ___, '#',  N ],
  [___, ___, ___, ___, '#', '#'],
  [DS , '#', '#', ___, '#', '#']
];

var mdp = makeGridWorldMDP({
  grid,
  noReverse: true,
  maxTimeAtRestaurant: 2,
  start: [3, 1],
  totalTime: 11
});

var donutSouthTrajectory = [
  [{"loc":[3,1],"terminateAfterAction":false,"timeLeft":11},"l"],
  [{"loc":[2,1],"terminateAfterAction":false,"timeLeft":10,"previousLoc":[3,1]},"l"],
  [{"loc":[1,1],"terminateAfterAction":false,"timeLeft":9,"previousLoc":[2,1]},"l"],
  [{"loc":[0,1],"terminateAfterAction":false,"timeLeft":8,"previousLoc":[1,1]},"d"],
  [{"loc":[0,0],"terminateAfterAction":false,"timeLeft":7,"previousLoc":[0,1],"timeAtRestaurant":0},"l"],
  [{"loc":[0,0],"terminateAfterAction":true,"timeLeft":7,"previousLoc":[0,0],"timeAtRestaurant":1},"l"]
];

var vegDirectTrajectory = [
  [{"loc":[3,1],"terminateAfterAction":false,"timeLeft":11},"u"],
  [{"loc":[3,2],"terminateAfterAction":false,"timeLeft":10,"previousLoc":[3,1]},"u"],
  [{"loc":[3,3],"terminateAfterAction":false,"timeLeft":9,"previousLoc":[3,2]},"u"],
  [{"loc":[3,4],"terminateAfterAction":false,"timeLeft":8,"previousLoc":[3,3]},"u"],
  [{"loc":[3,5],"terminateAfterAction":false,"timeLeft":7,"previousLoc":[3,4]},"u"],
  [{"loc":[3,6],"terminateAfterAction":false,"timeLeft":6,"previousLoc":[3,5]},"r"],
  [{"loc":[4,6],"terminateAfterAction":false,"timeLeft":5,"previousLoc":[3,6]},"u"],
  [{"loc":[4,7],"terminateAfterAction":false,"timeLeft":4,"previousLoc":[4,6],"timeAtRestaurant":0},"l"],
  [{"loc":[4,7],"terminateAfterAction":true,"timeLeft":4,"previousLoc":[4,7],"timeAtRestaurant":1},"l"]
];
///

var world = mdp.world;
var makeUtilityFunction = mdp.makeUtilityFunction;


// Priors

var utilityTablePrior = function() {
  var foodValues = [0, 1, 2];
  var timeCostValues = [-0.1, -0.3, -0.6];
  var donut = uniformDraw(foodValues);
  return {
    'Donut N': donut,
    'Donut S': donut,
    'Veg': uniformDraw(foodValues),
    'Noodle': uniformDraw(foodValues),
    'timeCost': uniformDraw(timeCostValues)
  };
};

var alphaPrior = function(){
  return uniformDraw([.1, 1, 10, 100]);
};


// Condition on observed trajectory

var posterior = function(observedTrajectory){
  return Infer({ model() {
    var utilityTable = utilityTablePrior();
    var alpha = alphaPrior();
    var params = {
      utility: makeUtilityFunction(utilityTable),
      alpha
    };
    var agent = makeMDPAgent(params, world);
    var act = agent.act;

    // For each observed state-action pair, factor on likelihood of action
    map(
      function(stateAction){
        var state = stateAction[0];
        var action = stateAction[1]
        observe(act(state), action);
      },
      observedTrajectory);

    // Compute whether Donut is preferred to Veg and Noodle
    var donut = utilityTable['Donut N'];
    var donutFavorite = (
      donut > utilityTable.Veg &&
      donut > utilityTable.Noodle);

    return {
      donutFavorite,
      alpha: alpha.toString(),
      timeCost: utilityTable.timeCost.toString()
    };
  }});
};

print('Prior:');
var prior = posterior([]);
viz.marginals(prior);

print('Conditioning on one action:');
var posterior = posterior(donutSouthTrajectory.slice(0, 1));
viz.marginals(posterior);
~~~~

## Example 4
<!-- display_multiple_trajectories -->
~~~~
///fold: make restaurant choice MDP, naiveTrajectory, donutSouthTrajectory
var ___ = ' ';
var DN = { name : 'Donut N' };
var DS = { name : 'Donut S' };
var V = { name : 'Veg' };
var N = { name : 'Noodle' };

var grid = [
  ['#', '#', '#', '#',  V , '#'],
  ['#', '#', '#', ___, ___, ___],
  ['#', '#', DN , ___, '#', ___],
  ['#', '#', '#', ___, '#', ___],
  ['#', '#', '#', ___, ___, ___],
  ['#', '#', '#', ___, '#',  N ],
  [___, ___, ___, ___, '#', '#'],
  [DS , '#', '#', ___, '#', '#']
];

var mdp = makeGridWorldMDP({
  grid,
  noReverse: true,
  maxTimeAtRestaurant: 2,
  start: [3, 1],
  totalTime: 11
});

var naiveTrajectory = [
  [{"loc":[3,1],"terminateAfterAction":false,"timeLeft":11},"u"],
  [{"loc":[3,2],"terminateAfterAction":false,"timeLeft":10,"previousLoc":[3,1]},"u"],
  [{"loc":[3,3],"terminateAfterAction":false,"timeLeft":9,"previousLoc":[3,2]},"u"],
  [{"loc":[3,4],"terminateAfterAction":false,"timeLeft":8,"previousLoc":[3,3]},"u"],
  [{"loc":[3,5],"terminateAfterAction":false,"timeLeft":7,"previousLoc":[3,4]},"l"],
  [{"loc":[2,5],"terminateAfterAction":false,"timeLeft":6,"previousLoc":[3,5],"timeAtRestaurant":0},"l"],
  [{"loc":[2,5],"terminateAfterAction":true,"timeLeft":6,"previousLoc":[2,5],"timeAtRestaurant":1},"l"]
];

var donutSouthTrajectory = [
  [{"loc":[3,1],"terminateAfterAction":false,"timeLeft":11},"l"],
  [{"loc":[2,1],"terminateAfterAction":false,"timeLeft":10,"previousLoc":[3,1]},"l"],
  [{"loc":[1,1],"terminateAfterAction":false,"timeLeft":9,"previousLoc":[2,1]},"l"],
  [{"loc":[0,1],"terminateAfterAction":false,"timeLeft":8,"previousLoc":[1,1]},"d"],
  [{"loc":[0,0],"terminateAfterAction":false,"timeLeft":7,"previousLoc":[0,1],"timeAtRestaurant":0},"l"],
  [{"loc":[0,0],"terminateAfterAction":true,"timeLeft":7,"previousLoc":[0,0],"timeAtRestaurant":1},"l"]
];
///

var world = mdp.world;;

map(function(trajectory) { viz.gridworld(world, { trajectory }); },
    [naiveTrajectory, donutSouthTrajectory]);
~~~~

To perform inference, we just condition on both sequences. (We use concatenation but we could have taken the union of all state-action pairs).

<!-- infer_from_multiple_trajectories -->
~~~~
///fold: World and agent are exactly as above

var ___ = ' ';
var DN = { name : 'Donut N' };
var DS = { name : 'Donut S' };
var V = { name : 'Veg' };
var N = { name : 'Noodle' };

var grid = [
  ['#', '#', '#', '#',  V , '#'],
  ['#', '#', '#', ___, ___, ___],
  ['#', '#', DN , ___, '#', ___],
  ['#', '#', '#', ___, '#', ___],
  ['#', '#', '#', ___, ___, ___],
  ['#', '#', '#', ___, '#',  N ],
  [___, ___, ___, ___, '#', '#'],
  [DS , '#', '#', ___, '#', '#']
];

var mdp = makeGridWorldMDP({
  grid,
  noReverse: true,
  maxTimeAtRestaurant: 2,
  start: [3, 1],
  totalTime: 11
});

var donutSouthTrajectory = [
  [{"loc":[3,1],"terminateAfterAction":false,"timeLeft":11},"l"],
  [{"loc":[2,1],"terminateAfterAction":false,"timeLeft":10,"previousLoc":[3,1]},"l"],
  [{"loc":[1,1],"terminateAfterAction":false,"timeLeft":9,"previousLoc":[2,1]},"l"],
  [{"loc":[0,1],"terminateAfterAction":false,"timeLeft":8,"previousLoc":[1,1]},"d"],
  [{"loc":[0,0],"terminateAfterAction":false,"timeLeft":7,"previousLoc":[0,1],"timeAtRestaurant":0},"l"],
  [{"loc":[0,0],"terminateAfterAction":true,"timeLeft":7,"previousLoc":[0,0],"timeAtRestaurant":1},"l"]
];


var naiveTrajectory = [
  [{"loc":[3,1],"terminateAfterAction":false,"timeLeft":11},"u"],
  [{"loc":[3,2],"terminateAfterAction":false,"timeLeft":10,"previousLoc":[3,1]},"u"],
  [{"loc":[3,3],"terminateAfterAction":false,"timeLeft":9,"previousLoc":[3,2]},"u"],
  [{"loc":[3,4],"terminateAfterAction":false,"timeLeft":8,"previousLoc":[3,3]},"u"],
  [{"loc":[3,5],"terminateAfterAction":false,"timeLeft":7,"previousLoc":[3,4]},"l"],
  [{"loc":[2,5],"terminateAfterAction":false,"timeLeft":6,"previousLoc":[3,5],"timeAtRestaurant":0},"l"],
  [{"loc":[2,5],"terminateAfterAction":true,"timeLeft":6,"previousLoc":[2,5],"timeAtRestaurant":1},"l"]
];

var world = mdp.world;
var makeUtilityFunction = mdp.makeUtilityFunction;


// Priors

var utilityTablePrior = function() {
  var foodValues = [0, 1, 2];
  var timeCostValues = [-0.1, -0.3, -0.6];
  var donut = uniformDraw(foodValues);
  return {
    'Donut N': donut,
    'Donut S': donut,
    'Veg': uniformDraw(foodValues),
    'Noodle': uniformDraw(foodValues),
    'timeCost': uniformDraw(timeCostValues)
  };
};

var alphaPrior = function(){
  return uniformDraw([.1, 1, 10, 100]);
};


// Condition on observed trajectory

var posterior = function(observedTrajectory){
  return Infer({ model() {
    var utilityTable = utilityTablePrior();
    var alpha = alphaPrior();
    var params = {
      utility: makeUtilityFunction(utilityTable),
      alpha
    };
    var agent = makeMDPAgent(params, world);
    var act = agent.act;

    // For each observed state-action pair, factor on likelihood of action
    map(
      function(stateAction){
        var state = stateAction[0];
        var action = stateAction[1]
        observe(act(state), action);
      },
      observedTrajectory);

    // Compute whether Donut is preferred to Veg and Noodle
    var donut = utilityTable['Donut N'];
    var donutFavorite = (
      donut > utilityTable.Veg &&
      donut > utilityTable.Noodle);

    return {
      donutFavorite,
      alpha: alpha.toString(),
      timeCost: utilityTable.timeCost.toString()
    };
  }});
};

///
print('Prior:');
var prior = posterior([]);
viz.marginals(prior);

print('Posterior');
var posterior = posterior(naiveTrajectory.concat(donutSouthTrajectory));
viz.marginals(posterior);
~~~~

## Example 5

~~~~
var inferBeliefsAndPreferences = function(baseAgentParams, priorPrizeToUtility,
                                          priorInitialBelief, bandit,
                                          observedSequence) {

  return Infer({ model() {

    // 1. Sample utilities
    var prizeToUtility = (priorPrizeToUtility ? sample(priorPrizeToUtility)
                          : undefined);

    // 2. Sample beliefs
    var initialBelief = sample(priorInitialBelief);

    // 3. Construct agent given utilities and beliefs
    var newAgentParams = extend(baseAgentParams, { priorBelief: initialBelief });
    var agent = makeBanditAgent(newAgentParams, bandit, 'belief', prizeToUtility);
    var agentAct = agent.act;
    var agentUpdateBelief = agent.updateBelief;

    // 4. Condition on observations
    var factorSequence = function(currentBelief, previousAction, timeIndex){
      if (timeIndex < observedSequence.length) {
        var state = observedSequence[timeIndex].state;
        var observation = observedSequence[timeIndex].observation;
        var nextBelief = agentUpdateBelief(currentBelief, observation, previousAction);
        var nextActionDist = agentAct(nextBelief);
        var observedAction = observedSequence[timeIndex].action;
        factor(nextActionDist.score(observedAction));
        factorSequence(nextBelief, observedAction, timeIndex + 1);
      }
    };
    factorSequence(initialBelief,'noAction', 0);

    return {
      prizeToUtility,
      priorBelief: initialBelief
    };
  }});
};
~~~~


## Example 6 

~~~~
///fold: inferBeliefsAndPreferences, getMarginal
var inferBeliefsAndPreferences = function(baseAgentParams, priorPrizeToUtility,
                                          priorInitialBelief, bandit,
                                          observedSequence) {

  return Infer({ model() {

    // 1. Sample utilities
    var prizeToUtility = (priorPrizeToUtility ? sample(priorPrizeToUtility)
                          : undefined);

    // 2. Sample beliefs
    var initialBelief = sample(priorInitialBelief);

    // 3. Construct agent given utilities and beliefs
    var newAgentParams = extend(baseAgentParams, { priorBelief: initialBelief });
    var agent = makeBanditAgent(newAgentParams, bandit, 'belief', prizeToUtility);
    var agentAct = agent.act;
    var agentUpdateBelief = agent.updateBelief;

    // 4. Condition on observations
    var factorSequence = function(currentBelief, previousAction, timeIndex){
      if (timeIndex < observedSequence.length) {
        var state = observedSequence[timeIndex].state;
        var observation = observedSequence[timeIndex].observation;
        var nextBelief = agentUpdateBelief(currentBelief, observation, previousAction);
        var nextActionDist = agentAct(nextBelief);
        var observedAction = observedSequence[timeIndex].action;
        factor(nextActionDist.score(observedAction));
        factorSequence(nextBelief, observedAction, timeIndex + 1);
      }
    };
    factorSequence(initialBelief,'noAction', 0);

    return {
      prizeToUtility,
      priorBelief: initialBelief
    };
  }});
};

var getMarginal = function(dist, key){
  return Infer({ model() {
    return sample(dist)[key];
  }});
};
///
// true prizes for arms
var trueArmToPrizeDist = {
  0: Delta({ v: 'chocolate' }),
  1: Delta({ v: 'champagne' })
};
var bandit = makeBanditPOMDP({
  armToPrizeDist: trueArmToPrizeDist,
  numberOfArms: 2,
  numberOfTrials: 5
});

// simpleAgent always pulls arm 1
var simpleAgent = makePOMDPAgent({
  act: function(belief){
    return Infer({ model() { return 1; }});
  },
  updateBelief: function(belief){ return belief; },
  params: { priorBelief: Delta({ v: bandit.startState }) }
}, bandit.world);

var observedSequence = simulatePOMDP(bandit.startState, bandit.world, simpleAgent,
                                    'stateObservationAction');

// Priors for inference

// We know agent's prior, which is that either arm1 yields
// nothing or it yields champagne.
var priorInitialBelief = Delta({ v: Infer({ model() {
  var armToPrizeDist = uniformDraw([
    trueArmToPrizeDist,
    extend(trueArmToPrizeDist, { 1: Delta({ v: 'nothing' }) })]);
  return makeBanditStartState(5, armToPrizeDist);
}})});

// Agent either prefers chocolate or champagne.
var likesChampagne = {
  nothing: 0,
  champagne: 5,
  chocolate: 3
};
var likesChocolate = {
  nothing: 0,
  champagne: 3,
  chocolate: 5
};
var priorPrizeToUtility = Categorical({
  vs: [likesChampagne, likesChocolate],
  ps: [0.5, 0.5]
});
var baseParams = { alpha: 1000 };
var posterior = inferBeliefsAndPreferences(baseParams, priorPrizeToUtility,
                                           priorInitialBelief, bandit,
                                           observedSequence);

print("After observing agent choose arm1, what are agent's utilities?");
print('Posterior on agent utilities:');
viz.table(getMarginal(posterior, 'prizeToUtility'));
~~~~
