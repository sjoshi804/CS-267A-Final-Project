"""An implementation of the pursuit-evasion game with a stationary evader with a gym environment wrapper using gym-pycolab."""

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


GAME_ART = ['#######################',
            '#P                   E#',            
            '#######################']
            

class OneStationaryEvaderEnv(gym_pycolab.PyColabEnv):
    """A gridworld based environment with 1 randomly moving evader. The agent 
    must catch the evader, reward = -1 for every timestep that the evader is 
    not caught and reward = 0 when evader is caught. Evader is 'caught' when
    agent is on a square adjacent to it. The environment uses the pycolab 
    gridworld game engine with a gym wrapper provide by gym-pycolab."""

    def __init__(self,
                 max_iterations=1000,
                 default_reward=-1.):
        super(OneStationaryEvaderEnv, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4))

    def make_game(self):
        """Builds and returns a four-rooms game with Pursuader and Evader."""
        return ascii_art.ascii_art_to_game(
            GAME_ART, what_lies_beneath=' ',
            sprites={'P': PursuerSprite, 'E': EvaderSprite}, update_schedule=[['E'], ['P']])

    def make_colors(self):
        return {'#': (0, 0, 255), 'P': (255, 0, 0), 'E': (0, 255, 0)}


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
    #Do nothing