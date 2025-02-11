#5
import pandas as pd
import os
from glob import glob

# Parameters
SELECTED_FEATURES_FILE = "../outputs/selected_features.csv"  # Input file containing features and correlations
TARGET_FOLDER = "../data/cleaned/train"          # Folder containing the cleaned training data
OUTPUT_FOLDER = "../data/cleaned/train_selected" # Folder to save the filtered dataset
CORRELATION_THRESHOLD = 0.05                     # Minimum absolute correlation to retain a feature
TARGET_COLUMN = "responder_6"                    # Target column for prediction

# Function to filter a DataFrame based on selected features
def filter_dataframe(df, selected_features):
    # Ensure the target column is included
    selected_features = list(selected_features) + [TARGET_COLUMN]
    print(f"Filtering DataFrame to retain features: {selected_features}")
    return df[selected_features]

# Load the selected features file
def load_selected_features(file_path, threshold):
    print(f"Loading selected features from {file_path}...")
    selected_features_df = pd.read_csv(file_path)
    selected_features = selected_features_df[
        selected_features_df["responder_6"].abs() > threshold
    ]["Unnamed: 0"].tolist()  # Use the feature names column
    print(f"Selected Features: {selected_features}")
    return selected_features

# Process partitioned Parquet files to create a cleaned dataset
def process_partitioned_data(input_folder, output_folder, selected_features, folder_pattern='partition_id=*'):
    # Find all Parquet files
    file_paths = glob(os.path.join(input_folder, f"{folder_pattern}/*.parquet"))
    print(f"Found {len(file_paths)} files in {input_folder} ({folder_pattern})")

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for file_path in file_paths:
        print(f"Processing: {file_path}")
        df = pd.read_parquet(file_path)

        # Filter the DataFrame
        df_filtered = filter_dataframe(df, selected_features)

        # Save the filtered dataset
        relative_path = os.path.relpath(file_path, input_folder)
        output_file_path = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # Create subdirectories if needed
        df_filtered.to_parquet(output_file_path, index=False)
        print(f"Filtered file saved to: {output_file_path}")

# Main Function
if __name__ == "__main__":
    # Load the selected features
    selected_features = load_selected_features(SELECTED_FEATURES_FILE, CORRELATION_THRESHOLD)

    # Process the cleaned training data
    print("Creating cleaned feature set for train data...")
    process_partitioned_data(
        input_folder=TARGET_FOLDER,
        output_folder=OUTPUT_FOLDER,
        selected_features=selected_features
    )
    print("Cleaned feature set creation complete!")
