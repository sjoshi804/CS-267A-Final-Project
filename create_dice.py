from math import exp as exp

# String Literals
LET = "let "
IN = "in"
IF = "if "
THEN = "then "
ELSE = " else "
DISCRETE = "discrete"
LBRACKET = "("
RBRACKET = ")"
FLIP = "flip "
FUNCTION = "fun"
INT = "int"
COMMA = ", "
NEWLINE = "\n"
EQUALS = " == "
AND = " && "
SPACE = " "
INDENT = "   "
ASSIGN_EQUAL = " = "
OBSERVE = "observe"
STATE = "STATE"
ACTION = "ACTION"
STATE_ = "STATE_"
ACTION_ = "ACTION_"

# Helper Functions
def let_expr(var, expression):
    return LET + str(var) + ASSIGN_EQUAL + NEWLINE + INDENT + str(expression) + SPACE + IN + NEWLINE

def if_expr(condition, then_expression, else_expression):
    return IF + str(condition) + NEWLINE + THEN + str(then_expression) + NEWLINE + ELSE + str(else_expression)

def if_expr_with_else_unspecified(condition, then_expression):
    return IF + str(condition) + NEWLINE + INDENT + THEN + str(then_expression) + ELSE + NEWLINE

def observe_expr(var):
    return let_expr("_", OBSERVE + LBRACKET + str(var) + RBRACKET)

def get_approx_dice_dist(distribution):
    new_dist = []
    string_dist = []
    #convert all but last element to strings
    for x in distribution[:-1]:
        string_dist.append(get_dice_prob(x))
        new_dist.append(float(get_dice_prob(x)))
    #set last element by calculating remaining probability mass - to ensure always adds up to 1
    remainder = 1
    for x in new_dist:
        remainder -= x
    string_dist.append(f'{remainder:9.8f}')
    #NOTE: Possibility of issues of distribution not normalized in dice due to overflow
    return string_dist
    

def discrete_distribution(distribution):
    #get stringified approximated dice distribution
    dice_dist = get_approx_dice_dist(distribution)
    expr = DISCRETE + LBRACKET + dice_dist[0]
    for x in dice_dist[1:]:
        expr += COMMA + SPACE + x
    
    expr += RBRACKET
    return expr

def flip_expr(prob_of_success):
    return FLIP + SPACE + get_dice_prob(prob_of_success)

def get_dice_type_of_int(size_of_space):
    return INT + LBRACKET + str(size_of_space) + RBRACKET

def get_dice_int(num, size_of_space):
    return INT + LBRACKET + str(size_of_space) + COMMA + str(num) + RBRACKET

def get_dice_prob(num):
    if num == 0:
        return "0.0"
    elif num == 1:
        return "1.0"
    else:
        return f'{num:9.7f}'

def equals_comparison(a, b):
    return str(a) + EQUALS + str(b)

def and_conditions(cond_1, cond_2):
    return str(cond_1) + AND + str(cond_2)

'''
Inputs:
- state space 
List of all states

- action space
List of all actions - transition function abstracts away idea of some actions not being avaiable in some states

- transition_function - takes in state and action returns a list of length = state_space which 
is list of probabilities to transitioning to the state at that index in state space

e.g. For 2 state example if state space = [A, B], action space = [1, 2, 3, 4], current state = A, current action = 1
s.t. 50% chance after this action we end up back in A and 50% chance we end up in B then
transition_function(A, 1) returns [0.5, 0.5]
'''
def translate_transition_function(transition_function, state_space, action_space):
    #Constructing Function Signature
    len_state_space = len(state_space)
    len_action_space = len(action_space)
    state_type = get_dice_type_of_int(len_state_space)
    action_type = get_dice_type_of_int(len_action_space)
    expr = FUNCTION + SPACE + "transition" + LBRACKET + "STATE: " + state_type + COMMA + "ACTION: " +  action_type + RBRACKET
    expr += NEWLINE + "{" + NEWLINE

    #Construct Function Body
    for state in state_space:
        for action in action_space:

            current_state = state_space.index(state)
            current_action = action_space.index(action)
            next_state_distribution = transition_function(state, action)

            state_condition = equals_comparison(STATE, get_dice_int(current_state, len_state_space))
            action_condition = equals_comparison("ACTION", get_dice_int(current_action, len_action_space))
            if_condition = and_conditions(state_condition, action_condition)
            then_expression = discrete_distribution(next_state_distribution)

            expr += if_expr_with_else_unspecified(if_condition, then_expression)
    
    # Adding extra else condition at end, this should never actually end up being used
    expr += get_dice_int(0, len_state_space)

    #Construct Function End
    expr += NEWLINE + "}"
    return expr + NEWLINE

def translate_reward_function(reward_function, state_space):
    #Constructing Function Signature
    len_state_space = len(state_space)
    state_type = get_dice_type_of_int(len_state_space)
    expr = FUNCTION + SPACE + "reward" + LBRACKET + "STATE: " + state_type + RBRACKET
    expr += NEWLINE + "{" + NEWLINE

    #Construct Function Body
    for state in state_space:
        prob_of_optimal = exp(reward_function(state))
        current_state = state_space.index(state)
          
        if_condition = equals_comparison(STATE, get_dice_int(current_state, len_state_space))
        then_expression = flip_expr(prob_of_optimal)

        expr += if_expr_with_else_unspecified(if_condition, then_expression)
    
    # Adding extra else condition at end, this should never actually end up being used
    expr += flip_expr(0)

    #Construct Function End
    expr += NEWLINE + "}"
    return expr + NEWLINE

#Call function with parameters STATE ACTION or just STATE if no action specified
def function_call(func_name, state_name, action_name=""):
    if action_name == "":
        return func_name + LBRACKET + state_name + RBRACKET
    else:
        return func_name + LBRACKET + state_name + COMMA + SPACE + action_name + RBRACKET

'''
Inputs:

- initial_state - distribution over initial states, specified as a list of length = state space
- max_length_of_trajectory - since dice can't do recursion, it can't handle infinite trajectories, hence
we must use finite trajectories, this is the maximum length of the trajectory
'''
def base_trajectory_distribution(max_length_of_trajectory, initial_state, action_prior):
    init_state_name = STATE_ + str(0)
    expr = let_expr(init_state_name, discrete_distribution(initial_state))
    current_state_name = init_state_name
    for t in range(0, max_length_of_trajectory):
        action_name = ACTION_ + str(t)
        expr += let_expr(action_name, discrete_distribution(action_prior))
        next_state_name = STATE_ + str(t + 1)
        expr += let_expr(next_state_name, function_call("transition", current_state_name, action_name))
        optimal_name = "OPTIMAL_" + str(t)
        expr += let_expr(optimal_name, function_call("reward", next_state_name))
        expr += observe_expr(optimal_name)
        current_state_name = next_state_name
    return expr

def trajectory_query(max_length_of_trajectory):
    expr = LBRACKET + ACTION_ + str(0)
    for t in range(1, max_length_of_trajectory):
        expr += COMMA + SPACE + ACTION_ + str(t)
    expr += RBRACKET
    return expr 

def next_action_query(state_space, current_state, current_step_number):
    current_state_dice = get_dice_int(state_space.index(current_state), len(state_space))
    expr = observe_expr(equals_comparison(STATE_ + str(current_step_number), get_dice_int(current_state, len(state_space))))
    expr += ACTION_ + str(current_step_number)
    return expr

def next_action_distribution(max_length_of_trajectory, initial_state, action_prior, state_space, current_state, current_step_number):
    return base_trajectory_distribution(max_length_of_trajectory, initial_state, action_prior) + next_action_query(state_space, current_state, current_step_number)

#DUMMY PROBLEM TO TEST - PURSUER IN TINY CORRIDOR AGAINST STATIONARY EVADER
def dummy_transition_function(state, action):
    distribution = [0] * 10
    if action == 0:
        if state != 0:
            distribution[state - 1] = 1
        else:
            distribution[state] = 1
    else:
        if state != 9:
            distribution[state + 1] = 1
        else:
            distribution[state] = 1
    return distribution

def dummy_reward_function(state):
    if state != 9:
        return -10
    else:
        return 0

dummy_state_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #Corridor [P         E]
dummy_action_space = [0, 1] #0 corresponds to LEFT and 1 corresponds to RIGHT
dummy_trajectory_length = 15
dummy_initial_state = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dummy_action_prior = [0.5, 0.5]

dice_reward_function = translate_reward_function(dummy_reward_function, dummy_state_space)
dice_transition_function = translate_transition_function(dummy_transition_function, dummy_state_space, dummy_action_space)
base_program = dice_reward_function + NEWLINE + dice_transition_function + NEWLINE + base_trajectory_distribution(dummy_trajectory_length, dummy_initial_state, dummy_action_prior)
print(base_program + next_action_query(dummy_state_space, 0, 0))