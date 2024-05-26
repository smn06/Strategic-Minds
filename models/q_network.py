import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Deep Q-network for reinforcement learning."""

    def __init__(self, state_size, action_size, seed=42):
        """Initialize parameters and build model.

        Params:
        - state_size (int): Dimension of each state
        - action_size (int): Dimension of each action
        - seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Forward pass through the network.

        Params:
        - state (tensor): Input state(s)

        Returns:
        - action_values (tensor): Predicted action values
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_values = self.fc3(x)
        return action_values
