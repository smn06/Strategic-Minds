import gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt

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
    def __init__(self, state_size, action_size, model_path):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_local.load_state_dict(torch.load(model_path))
        self.qnetwork_local.eval()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return np.argmax(action_values.cpu().data.numpy())

def run_inference(env, agent, n_episodes=100, max_t=1000):
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        print(f"Episode {i_episode}\tScore: {score:.2f}")
    return scores

def plot_scores(scores, filename):
    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(filename)

def main():
    env = gym.make('CartPole-v1')  # Example using CartPole environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model_path = 'checkpoint.pth'
    
    agent = DQNAgent(state_size, action_size, model_path)

    # Run inference
    scores = run_inference(env, agent)

    # Plot the scores
    plot_scores(scores, 'inference_scores.png')

if __name__ == "__main__":
    main()
