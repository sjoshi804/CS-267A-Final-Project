"""An implementation of the pursuit-evasion game in the classic four-rooms scenario with a gym environment wrapper using gym-pycolab."""

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


GAME_ART = ['#############',
            '#     #     #',
            '#     #     #',
            '#  E  #     #',
            '#           #',
            '#     #     #',
            '### ####### #',
            '#     #     #',
            '#     #     #',
            '#           #',
            '#     #     #',
            '# P   #     #',
            '#############']

class PursuitEvasionEnv(gym_pycolab.PyColabEnv):
    """A pycolab game env."""

    def __init__(self,
                 max_iterations=10,
                 default_reward=-1.):
        super(PursuitEvasionEnv, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4))

    def make_game(self):
        """Builds and returns a four-rooms game with Pursuader and Evader."""
        return ascii_art.ascii_art_to_game(
            GAME_ART, what_lies_beneath=' ',
            sprites={'P': PursuerSprite, 'E': EvaderSprite}, update_schedule=[['E'], ['P']])

    def make_colors(self):
        return {'#': (0, 0, 255)}




class PursuerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our pursuer.

  This `Sprite` ties actions to going in the four cardinal directions
  """

  def __init__(self, corner, position, character):
    """Inform superclass that we can't walk through walls."""
    super(PursuerSprite, self).__init__(
        corner, position, character, impassable='#')

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

    # See if we've found the mystery spot.
    if self.virtual_position == things['E'].virtual_position:
      the_plot.add_reward(100)
      the_plot.terminate_episode()
    else:
      the_plot.add_reward(-1) #Time cost

class EvaderSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our evader.

  This `Sprite` ties actions to going in the four cardinal directions. 
  """

  def __init__(self, corner, position, character):
    """Inform superclass that we can't walk through walls."""
    super(EvaderSprite, self).__init__(
        corner, position, character, impassable='#')

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



def main(argv=()):
    del argv  # Unused.

    # Build a four-rooms game.
    env = PursuitEvasionEnv()
    state = env.reset()

    # Random Agent playing as Pursuser
    while True:
  
        #Render environment #FIXME: issue with setup causing env.render() to fail on local machine
        env.render() #might need to run brew upgrade sdl2
        
        #Agent goes here - random agent for now
        action = env.action_space.sample() 

        #Get observations, rewards, termination form environment after taking action
        observation, reward, done, info = env.step(action) 
            
        if done: 
            break

    env.close()

if __name__ == '__main__':
  main(sys.argv)
