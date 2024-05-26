import numpy as np
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_reward = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, c_param=1.0):
        choices_weights = [
            (child.total_reward / child.visit_count) + c_param * math.sqrt((2 * math.log(self.visit_count) / child.visit_count))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def most_visited_child(self):
        return max(self.children, key=lambda child: child.visit_count)

class MCTS:
    def __init__(self, game, iterations=1000, c_param=1.0):
        self.game = game
        self.iterations = iterations
        self.c_param = c_param

    def search(self, initial_state):
        root = MCTSNode(initial_state)

        for _ in range(self.iterations):
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        return root.most_visited_child().state

    def _select(self, node):
        while not self.game.is_terminal(node.state):
            if node.is_fully_expanded():
                node = node.best_child(self.c_param)
            else:
                return self._expand(node)
        return node

    def _expand(self, node):
        tried_children = [child.state for child in node.children]
        new_state = self.game.next_state(node.state, random.choice(self.game.get_legal_actions(node.state)))
        while new_state in tried_children:
            new_state = self.game.next_state(node.state, random.choice(self.game.get_legal_actions(node.state)))
        child_node = MCTSNode(new_state, node)
        node.children.append(child_node)
        return child_node

    def _simulate(self, node):
        current_state = node.state
        while not self.game.is_terminal(current_state):
            action = random.choice(self.game.get_legal_actions(current_state))
            current_state = self.game.next_state(current_state, action)
        return self.game.get_reward(current_state)

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent
