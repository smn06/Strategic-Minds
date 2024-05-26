import gym
import numpy as np
import torch
import json
import random
import matplotlib.pyplot as plt

class CustomDota2Env(gym.Wrapper):
    def __init__(self, env):
        super(CustomDota2Env, self).__init__(env)
        # Custom initialization

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Custom processing
        return observation, reward, done, info

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
        self.qnetwork_local.eval()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return np.argmax(action_values.cpu().data.numpy())

def visualize_agent(env, agent, n_episodes=10, max_t=1000):
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            env.render()  # Render the environment
            score += reward
            if done:
                break
        print(f"Episode {i_episode}\tScore: {score:.2f}")

def main():
    env = gym.make('CartPole-v1')  # Example using CartPole environment
    env = CustomDota2Env(env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, seed=0)

    # Visualize the agent
    visualize_agent(env, agent)

    env.close()

if __name__ == "__main__":
    main()
