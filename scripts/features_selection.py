#4
import pandas as pd
import os
from glob import glob

# Parameters for feature selection
TARGET_COLUMN = "responder_6"  # Target responder for prediction
CORRELATION_THRESHOLD = 0.05   # Minimum absolute correlation to retain a feature
OUTPUT_SELECTED_FEATURES = "../outputs/selected_features.csv"

# Function to calculate feature correlations with the target
def select_features_based_on_correlation(df):
    print(f"Selecting features based on correlation with {TARGET_COLUMN}...")

    # Ensure the target column exists in the dataset
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} not found in the dataset.")

    # Select numeric features only
    numeric_features = df.select_dtypes(include="number").columns

    # Drop metadata or irrelevant columns if present
    excluded_columns = ["date_id", "time_id", "symbol_id", "weight"]
    numeric_features = [col for col in numeric_features if col not in excluded_columns]

    # Calculate correlations with the target
    correlations = df[numeric_features].corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
    print("Correlation Summary:")
    print(correlations)

    # Filter features based on the threshold
    selected_features = correlations[correlations.abs() > CORRELATION_THRESHOLD].index.tolist()
    print(f"Selected Features (Correlation > {CORRELATION_THRESHOLD}): {selected_features}")

    return selected_features, correlations

# Process partitioned Parquet files to aggregate feature selection
def process_partitioned_data(input_folder, folder_pattern='partition_id=*'):
    # Find all Parquet files
    file_paths = glob(os.path.join(input_folder, f"{folder_pattern}/*.parquet"))
    print(f"Found {len(file_paths)} files in {input_folder} ({folder_pattern})")

    aggregated_correlations = None
    all_selected_features = set()

    for file_path in file_paths:
        print(f"Processing: {file_path}")
        df = pd.read_parquet(file_path)

        # Select features based on correlation
        selected_features, correlations = select_features_based_on_correlation(df)

        # Aggregate selected features
        all_selected_features.update(selected_features)

        # Aggregate correlations across partitions
        if aggregated_correlations is None:
            aggregated_correlations = correlations
        else:
            aggregated_correlations = aggregated_correlations.add(correlations, fill_value=0)

    # Normalize aggregated correlations
    aggregated_correlations /= len(file_paths)

    return list(all_selected_features), aggregated_correlations

# Save selected features and correlations to CSV
def save_selected_features(features, correlations, output_path):
    print(f"Saving selected features to {output_path}...")
    correlations.loc[features].to_csv(output_path, index=True)

# Main Function
if __name__ == "__main__":
    print("Starting Feature Selection for Train Data...")
    selected_features, aggregated_correlations = process_partitioned_data(
        input_folder="../data/cleaned/train"
    )

    # Save the selected features and their correlations
    save_selected_features(selected_features, aggregated_correlations, OUTPUT_SELECTED_FEATURES)
    print(f"Feature selection complete. Results saved to {OUTPUT_SELECTED_FEATURES}.")
