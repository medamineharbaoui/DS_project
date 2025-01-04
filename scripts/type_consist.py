import pandas as pd
import os
from glob import glob
from tqdm import tqdm

def handle_types_and_missing_values(file_path_pattern, output_folder_path, folder_pattern='partition_id=*', handle_weights=True):
    """
    Function to handle type consistency and optionally missing value imputation for date_id, time_id, and weight columns
    in partitioned Parquet files.
    
    Args:
    - file_path_pattern: path to the folder containing partitioned Parquet files
    - output_folder_path: path to save the processed files
    - folder_pattern: partition pattern to match (default 'partition_id=*')
    - handle_weights: whether to handle missing values in the 'weight' column (default True)
    """
    # Ensure the output directory exists, create it if not
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Find all partitioned files using glob
    file_paths = glob(os.path.join(file_path_pattern, f'{folder_pattern}/*.parquet'))
    print(f"Found {len(file_paths)} files in {file_path_pattern} ({folder_pattern})")

    for file_path in tqdm(file_paths, desc="Processing files"):
        # Read the dataset
        print(f"Processing file: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Handle date_id - ensuring it is int16 (keeping it as ordinal identifier)
        print("Ensuring 'date_id' is of type int16...")
        df['date_id'] = df['date_id'].astype('int16')
        
        # Handle time_id - ensuring it is int16 (ordinal time index)
        print("Ensuring 'time_id' is of type int16...")
        df['time_id'] = df['time_id'].astype('int16')
        
        # Handle weight column if required
        if handle_weights:
            print("Handling missing values in 'weight'...")
            if df['weight'].isnull().sum() > 0:
                # Impute missing 'weight' with the median of the column
                median_weight = df['weight'].median()
                df['weight'].fillna(median_weight, inplace=True)
        
        # Construct the output file path preserving the folder structure
        relative_path = os.path.relpath(file_path, file_path_pattern)
        output_file_path = os.path.join(output_folder_path, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # Create subdirectories if needed
        
        print(f"Saving processed data to {output_file_path}...")
        df.to_parquet(output_file_path, index=False)
        print(f"Processing complete. File saved to {output_file_path}.")

# Example usage for processing Train, Test, and Lags data

# Processing train data (handle weight)
print("Processing Train Data:")
handle_types_and_missing_values('../data/train.parquet', '../data/preprocessed/train')

# Processing test data (handle weight)
print("Processing Test Data:")
handle_types_and_missing_values('../data/test.parquet', '../data/preprocessed/test', folder_pattern='date_id=*')

# Processing lags data (ignore weight)
print("Processing Lags Data:")
handle_types_and_missing_values('../data/lags.parquet', '../data/preprocessed/lags', folder_pattern='date_id=*', handle_weights=False)
