from math import exp as exp
import os 
import subprocess

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

class DiceProgramGenerator:
    """A tool to create dice programs corresponding to the RL as inference problem.

    Inputs: state_space, action_space, initial_state, action_prior, reward_function, transition_function

    Typically use method next_action(current_state, current_step_number) to infer next action

    """

    #Constructor
    def __init__(self, state_space, action_space, initial_state, action_prior,  reward_function, transition_function, max_trajectory_length):
        #Set instance variables
        self.state_space = state_space
        self.action_space = action_space
        self.initial_state = initial_state
        self.action_prior = action_prior
        self.reward_function = reward_function
        self.transition_function = transition_function
        self.max_trajectory_length = max_trajectory_length
    
        #Construct and cache copy of base program
        self.dice_reward_function = self.translate_reward_function(self.reward_function, self.state_space)
        self.dice_transition_function = self.translate_transition_function(self.transition_function, self.state_space, self.action_space)
        self.base_program = self.dice_reward_function + NEWLINE + self.dice_transition_function + NEWLINE + self.base_trajectory_distribution(self.max_trajectory_length, self.initial_state, self.action_prior)


    # Helper Functions
    def let_expr(self, var, expression):
        return LET + str(var) + ASSIGN_EQUAL + NEWLINE + INDENT + str(expression) + SPACE + IN + NEWLINE

    def if_expr(self, condition, then_expression, else_expression):
        return IF + str(condition) + NEWLINE + THEN + str(then_expression) + NEWLINE + ELSE + str(else_expression)

    def if_expr_with_else_unspecified(self, condition, then_expression):
        return IF + str(condition) + NEWLINE + INDENT + THEN + str(then_expression) + ELSE + NEWLINE

    def observe_expr(self, var):
        return self.let_expr("_", OBSERVE + LBRACKET + str(var) + RBRACKET)

    def get_approx_dice_dist(self, distribution):
        new_dist = []
        string_dist = []
        #convert all but last element to strings
        for x in distribution[:-1]:
            string_dist.append(self.get_dice_prob(x))
            new_dist.append(float(self.get_dice_prob(x)))
        #set last element by calculating remaining probability mass - to ensure always adds up to 1
        remainder = 1
        for x in new_dist:
            remainder -= x
        string_dist.append(f'{remainder:9.8f}')
        #NOTE: Possibility of issues of distribution not normalized in dice due to overflow
        return string_dist
        

    def discrete_distribution(self, distribution):
        #get stringified approximated dice distribution
        dice_dist = self.get_approx_dice_dist(distribution)
        expr = DISCRETE + LBRACKET + dice_dist[0]
        for x in dice_dist[1:]:
            expr += COMMA + SPACE + x
        
        expr += RBRACKET
        return expr

    def flip_expr(self, prob_of_success):
        return FLIP + SPACE + self.get_dice_prob(prob_of_success)

    def get_dice_type_of_int(self, size_of_space):
        return INT + LBRACKET + str(size_of_space) + RBRACKET

    def get_dice_int(self, num, size_of_space):
        return INT + LBRACKET + str(size_of_space) + COMMA + str(num) + RBRACKET

    def get_dice_prob(self, num):
        if num == 0:
            return "0.0"
        elif num == 1:
            return "1.0"
        else:
            return f'{num:9.7f}'

    def equals_comparison(self, a, b):
        return str(a) + EQUALS + str(b)

    def and_conditions(self, cond_1, cond_2):
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
    def translate_transition_function(self, transition_function=None, state_space=None, action_space=None):
        #Default Arguments
        if transition_function is None:
            transition_function = self.transition_function
        if state_space is None:
            state_space = self.state_space
        if action_space is None:
            action_space = self.action_space
        
        #Constructing Function Signature
        len_state_space = len(state_space)
        len_action_space = len(action_space)
        state_type = self.get_dice_type_of_int(len_state_space)
        action_type = self.get_dice_type_of_int(len_action_space)
        expr = FUNCTION + SPACE + "transition" + LBRACKET + "STATE: " + state_type + COMMA + "ACTION: " +  action_type + RBRACKET
        expr += NEWLINE + "{" + NEWLINE
        
        #Construct Function Body
        for state in state_space:
            for action in action_space:
                current_state = state_space.index(state)
                current_action = action_space.index(action)
                next_state_distribution = transition_function(state, action)

                state_condition = self.equals_comparison(STATE, self.get_dice_int(current_state, len_state_space))
                action_condition = self.equals_comparison("ACTION", self.get_dice_int(current_action, len_action_space))
                if_condition = self.and_conditions(state_condition, action_condition)
                then_expression = self.discrete_distribution(next_state_distribution)

                expr += self.if_expr_with_else_unspecified(if_condition, then_expression)

        # Adding extra else condition at end, this should never actually end up being used
        expr += self.get_dice_int(0, len_state_space)
        
        #Construct Function End
        expr += NEWLINE + "}"
        return expr + NEWLINE

    def translate_reward_function(self, reward_function=None, state_space=None):
        #Default Arguments
        if reward_function is None:
            reward_function = self.reward_function
        if state_space is None:
            state_space = self.state_space

        #Constructing Function Signature
        len_state_space = len(state_space)
        state_type = self.get_dice_type_of_int(len_state_space)
        expr = FUNCTION + SPACE + "reward" + LBRACKET + "STATE: " + state_type + RBRACKET
        expr += NEWLINE + "{" + NEWLINE

        #Construct Function Body
        for state in state_space:
            prob_of_optimal = exp(reward_function(state))
            current_state = state_space.index(state)
            
            if_condition = self.equals_comparison(STATE, self.get_dice_int(current_state, len_state_space))
            then_expression = self.flip_expr(prob_of_optimal)

            expr += self.if_expr_with_else_unspecified(if_condition, then_expression)
        
        # Adding extra else condition at end, this should never actually end up being used
        expr += self.flip_expr(0)

        #Construct Function End
        expr += NEWLINE + "}"
        return expr + NEWLINE

    #Call function with parameters STATE ACTION or just STATE if no action specified
    def function_call(self, func_name, state_name, action_name=""):
        if action_name == "":
            return func_name + LBRACKET + state_name + RBRACKET
        else:
            return func_name + LBRACKET + state_name + COMMA + SPACE + action_name + RBRACKET

    '''
    Inputs:

    - initial_state - distribution over initial states, specified as a list of length = state space
    - max_trajectory_length - since dice can't do recursion, it can't handle infinite trajectories, hence
    we must use finite trajectories, this is the maximum length of the trajectory
    '''
    def base_trajectory_distribution(self, max_trajectory_length=None, initial_state=None, action_prior=None):
        #Default Arguments
        if max_trajectory_length is None:
            max_trajectory_length=self.max_trajectory_length
        if initial_state is None:
            initial_state=self.initial_state
        if action_prior is None:
            action_prior=self.action_prior

        #Constructing Trajectory
        init_state_name = STATE_ + str(0)
        expr = self.let_expr(init_state_name, self.discrete_distribution(initial_state))
        current_state_name = init_state_name
        for t in range(0, max_trajectory_length):
            action_name = ACTION_ + str(t)
            expr += self.let_expr(action_name, self.discrete_distribution(action_prior))
            next_state_name = STATE_ + str(t + 1)
            expr += self.let_expr(next_state_name, self.function_call("transition", current_state_name, action_name))
            optimal_name = "OPTIMAL_" + str(t)
            expr += self.let_expr(optimal_name, self.function_call("reward", next_state_name))
            expr += self.observe_expr(optimal_name)
            current_state_name = next_state_name
        return expr

    def trajectory_query(self, max_trajectory_length=None):
        #Default Arguments
        if max_trajectory_length is None:
            max_trajectory_length=self.max_trajectory_length

        expr = LBRACKET + ACTION_ + str(0)
        for t in range(1, max_trajectory_length):
            expr += COMMA + SPACE + ACTION_ + str(t)
        expr += RBRACKET
        return expr 

    def legacy_next_action_query(self, current_state, current_step_number, state_space):
        current_state_dice = self.get_dice_int(state_space.index(current_state), len(state_space))
        expr = self.observe_expr(self.equals_comparison(STATE_ + str(current_step_number), current_state_dice))
        expr += ACTION_ + str(current_step_number)
        return expr

    def legacy_next_action(self, current_state, current_step_number):
        return self.base_program + self.legacy_next_action_query(current_state, current_step_number, self.state_space)

    def next_action(self, current_state):
        current_state_dice = self.get_dice_int(self.state_space.index(current_state), len(self.state_space))
        query = ACTION_ + str(0)
        return self.set_initial_state(self.base_program, current_state_dice) + ACTION_ + str(0)

    def set_initial_state(self, base_program, current_state_dice):
        STATE_0_ID = "let STATE_0 = "
        program = base_program.split(STATE_0_ID)
        head = program[0]
        tail = program[1][program[1].index("in"):]
        return head + STATE_0_ID + current_state_dice + SPACE + tail

class DiceInferenceEngine:
    """
    A tool to interface with dice - performs as an inferene engine to be used in python for the RL as inference problem

    Inputs: state_space, action_space, initial_state, action_prior, reward_function, transition_function, path_to_dice - specify path to dice binary e.g. ./Dice-macos.native

    Method: 
    reset -> resets step number to 0
    set_step(t) -> sets step to t - returns True if successful False if failed
    next(state) -> returns list of dimension equal to action_space with probabilities for each action for the next step, None if step_num >= max_trajectory_length
    Helper: parse_output(dice_output) parses dice output and returns python list
    """
    #Constructor
    def __init__(self, state_space, action_space, initial_state, action_prior,  reward_function, transition_function, max_trajectory_length, path_to_dice="./Dice-macos.native"):
        self.program = DiceProgramGenerator(state_space, action_space, initial_state, action_prior, reward_function, transition_function, max_trajectory_length)
        self.step_num = 0
        self.path_to_dice = path_to_dice
        self.max_trajectory_length = max_trajectory_length
        self.trajectory = []

    #Reset execution to beginning
    def reset(self):
        self.step_num = 0
    
    #Set step number
    def set_step(self, step_num):
        if step_num < self.max_trajectory_length and step_num >= 0:
            self.step_num = step_num
            return True
        else:
            return False

    def next(self, state):
        #Check if steps left
        if self.step_num >= self.max_trajectory_length:
            print("Inference complete.")
            return None

        #Append states to trajectory
        self.trajectory.append(state)

        #Check if directory to store inference file exists or not
        if not os.path.exists('.infer'):
            os.mkdir('.infer')
        
        #Create dice program and execute it
        with open(".infer/step_" + str(self.step_num) + ".dice", "w") as dice_file:
            dice_file.write(self.program.next_action(state))
        cmd = self.path_to_dice + " " + ".infer/step_" + str(self.step_num) + ".dice"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        
        # If error executing then print error and return None
        if not error is None:
            print(error.decode("utf-8"))
            return None
        
        # If error in dice program, print error and return none
        if output.decode("utf-8")[0:5] != "Value":
            print(output.decode("utf-8"))
            return None
        
        #Increment step num
        self.step_num += 1
        
        
        # Parse Output and Return List
        return self.parse_output(output.decode("utf-8"))
    
    def parse_output(self, dice_output):
        dice_output = dice_output.splitlines()[1:]
        list_of_prob = []
        for line in dice_output:
            if line[0] == 'F': #check if we have reached line Final compiled size
                break
            list_of_prob.append(float(line.split("\t")[1]))
        return list_of_prob
    

""" #DUMMY PROBLEM TO TEST - PURSUER IN TINY CORRIDOR AGAINST STATIONARY EVADER
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
        return -2
    else:
        return 0

dummy_state_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #Corridor [P         E]
dummy_action_space = [0, 1] #0 corresponds to LEFT and 1 corresponds to RIGHT
dummy_max_trajectory_length = 15
dummy_initial_state = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dummy_action_prior = [0.5, 0.5]

dice = DiceInferenceEngine(dummy_state_space, dummy_action_space, dummy_initial_state, dummy_action_prior,  dummy_reward_function, dummy_transition_function, dummy_max_trajectory_length)

state = 0
for x in range(0, 10):
    print(dice.next(state))
    state += 1 """