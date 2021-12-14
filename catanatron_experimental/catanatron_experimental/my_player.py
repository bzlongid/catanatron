from typing import Iterable

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player

import math
import queue


# NODE CLASS & IMPLEMENTATION BASED OFF LAB 7: MCTS

import numpy as np
example_board = np.array([[' ', ' ', ' '],
                          [' ', ' ', ' '],
                          [' ', ' ', ' ']])

# Calculates all successor states to a given state
def get_possible_moves(board, player):
    moves = []
    for (x, y), element in np.ndenumerate(board):
        if element == ' ':
            new_board = np.array(board, copy=True)
            new_board[x][y] = 'X' if player is 'max' else 'O'
            moves.append(new_board)
    return moves

# Hash function that returns a unique string for each board
def get_hash(board):
   if board is None:
     return "None"
   h = "" 
   for element in board.flatten():
     if element == ' ':
       h += '_'
     else:
       h += element
   return h
  
# Returns the value of a board: 
#  1 for a win for max (optionally adjusted by depth)
#  - for a win for min
#  0 for draws
#  None for non-terminal states
def get_score(board, depth=0):
    if (np.any(np.all(board == 'X', axis=0)) or 
        np.any(np.all(board == 'X', axis=1)) or 
        np.all(board.diagonal() == 'X') or 
        np.all(np.fliplr(board).diagonal() == 'X')):
        # Max Victory
        return 1 * (1 / (1 + depth))
    elif (np.any(np.all(board == 'O', axis=0)) or 
          np.any(np.all(board == 'O', axis=1)) or
          np.all(board.diagonal() == 'O') or 
          np.all(np.fliplr(board).diagonal() == 'O')):
        # Min Victory
        return -1 * (1 / (1 + depth))
    elif not (board == ' ').any():
        # Draw
        return 0
    else:
        # Unfinished Game
        return None

# Returns the next player to act
def next_player(player):
  if player=='max':
    return 'min'
  elif player== 'min':
    return 'max'
  else:
    raise ValueError('player must be max or min')


class Node:
  
    def __init__(self, board, parent, player='max'):
        self.min_wins = 0
        self.max_wins = 0
        self.board = board
        self.parent = parent
        self.count = 0
        self.children = {}
        self.player = player
    
    def __str__(self):
      if self.parent is None:
        p = "None"
      else:
        p = get_hash(self.parent.board)
      try:
        expected_value = self.get_expected_value()
      except ValueError:
        expected_value = 0
      s = "Node {}\n with parent {}\n Count = {}\n Max Wins = {}\n Min Wins = {}\n Expected Value = {}\n UCB = {}".format(hash(self.board),p,self.count,self.max_wins,self.min_wins,expected_value,self.get_ucb(next_player(self.player)))
      return s
      # TODO: look into self.getucb not passing c

    def add_child(self, child_board, player):
      key = get_hash(child_board)
      if key in self.children.keys():
        raise ValueError('child already exists')
      else:
        self.children[key] = Node(child_board,self,player)
        return self.children[key]
       
    def get_p_win(self, player):
        try:
            if player == 'min':
                return self.min_wins / self.count
            elif player == 'max':
                return self.max_wins / self.count
            else:
                raise ValueError('player {} must be min or max'.format(player))
        except ZeroDivisionError:
            raise ValueError('must be updated at least once \
                              to get win probability')

    def get_expected_value(self):
        try:
            return (self.max_wins -self.min_wins) / self.count
        except ZeroDivisionError:
            raise ValueError('must be updated at least once \
                              to get expected value')

    def get_explore_term(self, parent, c=1):
        if self.parent is not None:
            return c * (2* math.log(parent.count) / self.count) ** (1 / 2)
        else:
            return 0 
        
        
    def get_ucb(self, c=1, default=6):
        if self.count:
            p_win = self.get_expected_value()
            # This next step is a bit unintuivite: since a child playing "max" will be expanded by a parent who is playing "min",
            # the parent will want to expand a child with low (negative) expected value.
            # By reversing the sign here, we make it so whichever method is is calling get_ucb never has to worry about players and can always aim to maximize it.
            if self.player == "max":
              p_win *=-1
            explore_term = self.get_explore_term(self.parent,c)
            return p_win + explore_term
        else:
            return default

# MyPlayer is the Player which will implement MCTS to search the best action

class MyPlayer(Player):
    
    def expand(node, player):
        for successor in get_possible_moves(node.board, player):
            child = None
        try:
            child = node.add_child(successor.next_player(player))
        except ValueError: 
        # Guards against expanding the same child multiple times
            pass
        return child

    def tree_policy(self, node, player):
        while get_score(node.board) == None:
            if len(node.children) < len(get_possible_moves(node.board, player)):
                return self.expand(node, player)
            else:
                node = self.tree_policy(self.best_child(node), next_player(player))
        return node

    def best_child(node,c=1):
        best = node
        for child_key in node.children:
            child_node = node.children[child_key]
        best_ucb = max(child_node.get_ucb(c), best.get_ucb(c))
        if best_ucb != best.get_ucb(c):
            best = child_node
        return best

    def backup(self, node, score):
        # base case:
        if node.parent == None:
            return None

        if score > 0:
            node.max_wins += 1
        elif score < 0:
            node.min_wins += 1
        node.count += 1
        self.backup(node.parent, score)

    def default_policy(node):
        while get_score(node.board) == None:
            next_moves = get_possible_moves(node.board, next_player(node.player))
            node.board = np.random.choice(next_moves)
        score = get_score(node.board)
        final_board = node.board
        return score

    def decide(self, game: Game, playable_actions: Iterable[Action]):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # ===== YOUR CODE HERE =====
        # As an example we simply return the first action:
        # return playable_actions[0]
        start_node = Node(Game,None,'max')
        for iteration in range(25): # we chose 25 in the interest of time
            v = self.tree_policy(start_node, 'max')
            value = self.default_policy(v)
            self.backup(v,value)
        action = self.best_child(start_node,0).board
        return action
        # ===== END YOUR CODE =====
