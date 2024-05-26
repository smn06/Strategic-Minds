import unittest
import gym
import torch
import numpy as np
from src.models.dqn_agent import QNetwork, DQNAgent, ReplayBuffer

class TestQNetwork(unittest.TestCase):
    def setUp(self):
        self.state_size = 4
        self.action_size = 2
        self.qnetwork = QNetwork(self.state_size, self.action_size)

    def test_forward_pass(self):
        state = torch.randn(1, self.state_size)
        action_values = self.qnetwork(state)
        self.assertEqual(action_values.shape, (1, self.action_size))

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = DQNAgent(self.state_size, self.action_size, seed=0)

    def test_act(self):
        state = np.random.random(self.state_size)
        action = self.agent.act(state)
        self.assertIn(action, range(self.action_size))

    def test_step(self):
        state = self.env.reset()
        action = self.agent.act(state)
        next_state, reward, done, _ = self.env.step(action)
        self.agent.step(state, action, reward, next_state, done)
        self.assertIsInstance(self.agent.memory, ReplayBuffer)

    def test_learn(self):
        state = self.env.reset()
        for _ in range(10):
            action = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.step(state, action, reward, next_state, done)
            state = next_state
        self.assertGreater(len(self.agent.memory), 0)

if __name__ == '__main__':
    unittest.main()
