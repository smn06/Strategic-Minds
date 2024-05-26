import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file containing the data.

    Returns:
    - pd.DataFrame: Loaded data as a Pandas DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_features(data):
    """
    Perform feature preprocessing.

    Parameters:
    - data (pd.DataFrame): Input data with columns to be processed.

    Returns:
    - pd.DataFrame: Processed features ready for modeling.
    """
    # Example preprocessing steps (customize as per data characteristics)
    
    # Drop unnecessary columns
    data = data.drop(columns=['unnecessary_column1', 'unnecessary_column2'])
    
    # Normalize or scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['feature1', 'feature2', 'feature3']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # Perform one-hot encoding for categorical variables
    # Example:
    # data = pd.get_dummies(data, columns=['categorical_feature'])
    
    return data

def save_processed_data(data, output_file):
    """
    Save processed data to a CSV file.

    Parameters:
    - data (pd.DataFrame): Processed data to be saved.
    - output_file (str): File path to save the processed data.
    """
    data.to_csv(output_file, index=False)

def main():
    # Step 1: Load data
    data_path = 'data/raw/raw_dataset.csv'
    data = load_data(data_path)

    # Step 2: Preprocess features
    processed_data = preprocess_features(data)

    # Step 3: Save processed data
    output_file = 'data/processed/processed_dataset.csv'
    save_processed_data(processed_data, output_file)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()
