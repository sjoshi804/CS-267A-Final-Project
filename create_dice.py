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

# Helper Functions
def let_statement(var, expression):
    return (LET + var + ASSIGN_EQUAL + NEWLINE + INDENT + expression + NEWLINE + IN)

def if_statement(condition, then_expression, else_expression):
    return IF + condition + NEWLINE + THEN + then_expression + NEWLINE + ELSE + else_expression

def if_statement_with_else_unspecified(condition, then_expression):
    return IF + condition + NEWLINE + INDENT + THEN + then_expression + ELSE + NEWLINE

def discrete_distribution(distribution):
    statement = DISCRETE + LBRACKET + str(distribution[0])
    for x in distribution[1:]:
        statement += COMMA + SPACE + str(x)
    statement += RBRACKET
    return statement

def flip_statement(prob_of_success):
    statement = FLIP
    if prob_of_success == 0:
        statement += "0.0"
    elif prob_of_success == 1:
        statement += "1.0"
    else:
        statement += str(prob_of_success)
    return statement

def get_dice_type_of_int(size_of_space):
    return INT + LBRACKET + str(size_of_space) + RBRACKET

def get_dice_int(num, size_of_space):
    return INT + LBRACKET + str(size_of_space) + COMMA + str(num) + RBRACKET

def equals_comparison(a, b):
    return a + EQUALS + b

def and_conditions(cond_1, cond_2):
    return cond_1 + AND + cond_2

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
    statement = FUNCTION + SPACE + "reward" + LBRACKET + "STATE: " + state_type + COMMA + "ACTION: " +  action_type + RBRACKET
    statement += NEWLINE + "{" + NEWLINE

    #Construct Function Body
    for state in state_space:
        for action in action_space:

            current_state = state_space.index(state)
            current_action = action_space.index(action)
            next_state_distribution = transition_function(state, action)

            state_condition = equals_comparison("STATE", get_dice_int(current_state, len_state_space))
            action_condition = equals_comparison("ACTION", get_dice_int(current_action, len_action_space))
            if_condition = and_conditions(state_condition, action_condition)
            then_expression = discrete_distribution(next_state_distribution)

            statement += if_statement_with_else_unspecified(if_condition, then_expression)
    
    # Adding extra else condition at end, this should never actually end up being used
    statement += get_dice_int(0, len_state_space)

    #Construct Function End
    statement += NEWLINE + "}"
    return statement

def dummy_transition_function(state, action):
    return [0.5, 0.5]

state_space = [0, 1]
action_space = [0, 1, 2]

print(translate_transition_function(dummy_transition_function, state_space, action_space))