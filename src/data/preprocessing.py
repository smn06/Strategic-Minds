import gym
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Custom environment wrapper (example)
class CustomDota2Env(gym.Wrapper):
    def __init__(self, env):
        super(CustomDota2Env, self).__init__(env)
        # Custom initialization

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Custom processing
        return observation, reward, done, info

# Function to collect data
def collect_data(env, num_episodes):
    data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Replace with actual policy
            next_state, reward, done, _ = env.step(action)
            data['states'].append(state)
            data['actions'].append(action)
            data['rewards'].append(reward)
            data['next_states'].append(next_state)
            data['dones'].append(done)
            state = next_state
    
    return data

# Function to clean data
def clean_data(data):
    cleaned_data = {
        key: [entry for entry in value if entry is not None]
        for key, value in data.items()
    }
    return cleaned_data

# Function to normalize data
def normalize_data(data):
    states = np.array(data['states'])
    normalized_states = (states - np.mean(states, axis=0)) / np.std(states, axis=0)
    data['states'] = normalized_states.tolist()
    return data

# Function for feature engineering
def feature_engineering(data):
    cumulative_rewards = np.cumsum(data['rewards'])
    data['cumulative_rewards'] = cumulative_rewards.tolist()
    return data

# Function to split data
def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    return train_data, val_data, test_data

# Function to save data
def save_data(train_data, val_data, test_data):
    with open('processed_train_data.json', 'w') as f:
        json.dump(train_data, f)
    with open('processed_val_data.json', 'w') as f:
        json.dump(val_data, f)
    with open('processed_test_data.json', 'w') as f:
        json.dump(test_data, f)

# Main function to execute all steps
def main():

    env = gym.make('Dota2-v0')  # Example for a hypothetical Dota 2 environment
    env = CustomDota2Env(env)


    num_episodes = 100  # Number of episodes for data collection
    data = collect_data(env, num_episodes)
    
    # Save raw collected data
    with open('raw_game_data.json', 'w') as f:
        json.dump(data, f)


    data = clean_data(data)
    data = normalize_data(data)
    data = feature_engineering(data)
    train_data, val_data, test_data = split_data(data)
    
    # Save processed data
    save_data(train_data, val_data, test_data)

if __name__ == "__main__":
    main()
