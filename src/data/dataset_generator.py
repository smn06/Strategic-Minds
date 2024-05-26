import numpy as np
import json
from sklearn.model_selection import train_test_split

# Sample dataset generator
def generate_sample_data(num_samples):
    # Define the size of the state space and action space
    state_size = 10  # Example state size
    action_size = 5  # Example action size
    
    # Initialize the data structure
    data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': []
    }
    
    for _ in range(num_samples):
        state = np.random.rand(state_size).tolist()
        action = np.random.randint(0, action_size)
        reward = np.random.rand()
        next_state = np.random.rand(state_size).tolist()
        done = np.random.choice([True, False])
        
        data['states'].append(state)
        data['actions'].append(action)
        data['rewards'].append(reward)
        data['next_states'].append(next_state)
        data['dones'].append(done)
    
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

# Main function to generate and process sample data
def main():
    num_samples = 10000  # Number of samples to generate
    
    # Generate sample data
    data = generate_sample_data(num_samples)
    
    # Save raw generated data
    with open('raw_sample_data.json', 'w') as f:
        json.dump(data, f)
    
    # Step 3.3: Data Preprocessing
    data = clean_data(data)
    data = normalize_data(data)
    data = feature_engineering(data)
    train_data, val_data, test_data = split_data(data)
    
    # Save processed data
    save_data(train_data, val_data, test_data)

if __name__ == "__main__":
    main()
