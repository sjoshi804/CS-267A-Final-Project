"""An implementation of the pursuit-evasion game with a random evader with a gym environment wrapper using gym-pycolab."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import sys
from random import randint as randint
from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites
import gym_pycolab
from gym import spaces
from copy import deepcopy as deepcopy


GAME_ART = ['#######',
            '#P    #',
            '#    E#',
            '#######'] 

time_cost = -10
goal_reward = 0
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class OneRandomEvaderEnv(gym_pycolab.PyColabEnv):
    """A gridworld based environment with 1 randomly moving evader. The agent 
    must catch the evader, reward = -10 for every timestep that the evader is 
    not caught and reward = 0 when evader is caught. Evader is 'caught' when
    agent is on a square adjacent to it. The environment uses the pycolab 
    gridworld game engine with a gym wrapper provide by gym-pycolab."""

    def __init__(self,
                 max_iterations=1000,
                 default_reward=-1.):
        super(OneRandomEvaderEnv, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4))
    def get_initial_state(self):
      board = []
      for row in GAME_ART:
        board.append(list(row))
      return board

    def make_game(self):
        """Builds and returns a game with Pursuader and Evader."""
        return ascii_art.ascii_art_to_game(
            GAME_ART, what_lies_beneath=' ',
            sprites={'P': PursuerSprite, 'E': EvaderSprite}, update_schedule=[['E'], ['P']])

    def make_colors(self):
        return {'#': (0, 0, 255), 'P': (255, 0, 0), 'E': (0, 255, 0)}

    def get_reward_function(self):
      
      #Define reward function
      def reward_function(observed_state):
        #copyin global parameters to avoid referencing issues
        time_cost = -10
        goal_reward = 0

        #check if goal state
        is_goal_state = False
        for row in range(0, len(observed_state)):
          
          #if P not found then continue
          try:
            col = observed_state[row].index('P')
          except: 
            continue

          #Check if evader to top
          if row > 0 and observed_state[row - 1][col] == 'E':
            is_goal_state = True
            break
          #Check if evader to bottom
          if row < len(observed_state) - 1 and observed_state[row + 1][col] == 'E':
            is_goal_state = True
            break
          #Check if evader to left
          if col > 0 and observed_state[row][col - 1] == 'E':
            is_goal_state = True
            break
          #Check if evader to bottom
          if col < len(observed_state[row]) - 1 and observed_state[row][col + 1] == 'E':
            is_goal_state = True
            break
        
        #if goal state return goal reward
        if is_goal_state:
          return goal_reward
        #else return time cost
        else: 
          return time_cost
      
      return reward_function
    
    def get_transition_function(self):

      #Define transition function
      def transition_function(observed_state, action):

        #copies current state into next state
        next_state = deepcopy(observed_state)

        #see if action valid and if so return new state
        for row in range(0, len(observed_state)):

          #if P not found then continue
          try:
            col = observed_state[row].index('P')
          except: 
            continue

          #If move is UP and possible 
          if action == UP and row > 0 and observed_state[row-1][col] == ' ':
            next_state[row][col] = observed_state[row-1][col]
            next_state[row-1][col] = 'P'
            break
          #If move is DOWN and possible 
          if action == DOWN and row < len(observed_state) - 1 and observed_state[row+1][col] == ' ':
            next_state[row][col] = observed_state[row+1][col]
            next_state[row+1][col] = 'P'
            break
          #If move is LEFT and possible 
          if action == LEFT and col > 0 and observed_state[row][col-1] == ' ':
            next_state[row][col] = observed_state[row][col-1]
            next_state[row][col-1] = 'P'
            break
          #If move is RIGHT and possible 
          if action == RIGHT and col < len(observed_state[row]) - 1 and observed_state[row][col+1] == ' ':
            next_state[row][col] = observed_state[row][col+1]
            next_state[row][col+1] = 'P'
            break

        #Create four possible states that could result from the four possible moves the evader could take
        next_state_up = deepcopy(next_state)
        next_state_down = deepcopy(next_state)
        next_state_left = deepcopy(next_state)
        next_state_right = deepcopy(next_state)

        #Find all states that the evader may take us to
        for row in range(0, len(next_state)):

          #if P not found then continue
          try:
            col = next_state[row].index('E')
          except: 
            continue

          #If UP is possible - change next state up to be the state after evader moving UP
          if row > 0 and next_state[row-1][col] == ' ':
            next_state_up[row][col] = next_state[row-1][col]
            next_state[row-1][col] = 'E'
            break

          #If DOWN is possible - change next_state_down to be the state after evader moving DOWN
          if row < len(next_state) - 1 and next_state[row+1][col] == ' ':
            next_state_down[row][col] = next_state[row+1][col]
            next_state_down[row+1][col] = 'E'
            break

          #If LEFT is possible - change next_state_left to be the state after evader moving LEFT
          if col > 0 and next_state[row][col-1] == ' ':
            next_state_left[row][col] = next_state[row][col-1]
            next_state_left[row][col-1] = 'E'
            break

          #If RIGHT is possible - change next_state_right to be the state after evader moving LEFT
          if col < len(next_state[row]) - 1 and next_state[row][col+1] == ' ':
            next_state_right[row][col] = next_state[row][col+1]
            next_state_right[row][col+1] = 'E'
            break

        #Return list of probabilities of transition to each state in observed_state_space
        def prob_next_state(state):
          count = 0
          if state == next_state_up:
            count += 1
          if state == next_state_down:
            count += 1
          if state == next_state_left:
            count += 1
          if state == next_state_right:
            count += 1
          return count * 0.25

        return [prob_next_state(state) for state in self.get_observed_state_space()]
      
      return transition_function

    def get_observed_state_space(self):

      #Helper Functions
      def remove_agents(art):
        #Define function that takes agent and returns blank else does nothing
        def set_agent_to_blank(square):
          if square == 'P' or square == 'E':
            return ' '
          else:
            return square

        board = []
        for row in art:
          ##print(list(row))
          board.append([set_agent_to_blank(square) for square in list(row)])
        return board

      #Constructed observed state space
      observed_state_space = []
      art = remove_agents(GAME_ART) #removes agents and makes the art into a list of lists - grid like

      #For all positios of pursuer
      for pursuer_row in range(0, len(art)):
        for pursuer_col in range(0, len(art[pursuer_row])):
          #Place Pursuer
          new_art = deepcopy(art)
          if new_art[pursuer_row][pursuer_col] != ' ': #can't if square is not empty
            continue 
          new_art[pursuer_row][pursuer_col] = 'P'

          #For all positions of evader
          for evader_row in range(0, len(new_art)):
            for evader_col in range(0, len(new_art[evader_row])):
              #Place Evader
              if new_art[evader_row][evader_col] != ' ': #can't if square is not empty
                continue 
              new_new_art = deepcopy(new_art)
              new_new_art[evader_row][evader_col] = 'E'

              #Color board and add to observed state space
              observed_state_space.append(new_new_art)
      return observed_state_space

    def uncolor_board(self, board):
      #Define uncoloring function
      def uncolor_square(square):
        #FIXME: Make this general - tried with reversing key, value but run into type issues
        if square[0] == 0 and square[1] == 0 and square[2] == 255:
          return '#'
        elif square[0] == 0 and square[1] == 255 and square[2] == 0:
          return 'E'
        elif square[0] == 255 and square[1] == 0 and square[2] == 0:
          return 'P'
        else:
          return ' '

      art = []
      for row in board:
        art.append([uncolor_square(square) for square in row])
       
      return art    
class PursuerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our pursuer.

  This `Sprite` ties actions to going in the four cardinal directions
  """

  def __init__(self, corner, position, character):
    """Inform superclass that we can't walk through walls or other agents."""
    super(PursuerSprite, self).__init__(
        corner, position, character, impassable='#E')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del layers, backdrop   # Unused.
  
    # Apply motion commands.
    if actions == 0:    # walk upward?
      self._north(board, the_plot)
    elif actions == 1:  # walk downward?
      self._south(board, the_plot)
    elif actions == 2:  # walk leftward?
      self._west(board, the_plot)
    elif actions == 3:  # walk rightward?
      self._east(board, the_plot)

    # See if we are adjacent to evader
    self_row = self.virtual_position.row
    self_col = self.virtual_position.col
    evader_row = things['E'].virtual_position.row
    evader_col = things['E'].virtual_position.col
    up_pos = (self_row - 1 == evader_row) and (self_col == evader_col)
    down_pos = (self_row + 1 == evader_row) and (self_col == evader_col)
    left_pos = (self_row == evader_row) and (self_col -1 == evader_col)
    right_pos = (self_row == evader_row) and (self_col + 1 == evader_col)
    if up_pos or down_pos or left_pos or right_pos:
      the_plot.add_reward(0)
      the_plot.terminate_episode()
    else:
      the_plot.add_reward(-1) #Time cost

class EvaderSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our evader.

  This `Sprite` ties actions to going in the four cardinal directions. 
  """

  def __init__(self, corner, position, character):
    """Inform superclass that we can't walk through walls or or other agents."""
    super(EvaderSprite, self).__init__(
        corner, position, character, impassable='#P')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del layers, backdrop, things   # Unused.

    random_action = randint(0, 3)

    # Apply motion commands.
    if random_action == 0:    # walk upward?
      self._north(board, the_plot)
    elif random_action == 1:  # walk downward?
      self._south(board, the_plot)
    elif random_action == 2:  # walk leftward?
      self._west(board, the_plot)
    elif random_action == 3:  # walk rightward?
      self._east(board, the_plot) 