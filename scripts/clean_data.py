#3
import pandas as pd
import os
from glob import glob

# Parameters for cleaning
COLUMNS_TO_REMOVE = [
    "feature_00", "feature_01", "feature_02", "feature_03", "feature_04",  # 100% missing features
    "feature_21", "feature_26", "feature_27", "feature_31"                # High missing values
]
MISSING_VALUE_THRESHOLD = 0.2  # Remove features with >20% missing values
IMPUTATION_METHOD = "median"   # Options: "mean", "median"

# Function to clean a single DataFrame
def clean_dataframe(df):
    # Remove predefined columns
    print(f"Removing columns: {COLUMNS_TO_REMOVE}")
    df.drop(columns=[col for col in COLUMNS_TO_REMOVE if col in df.columns], inplace=True, errors="ignore")
    
    # Identify columns exceeding the missing value threshold
    missing_ratios = df.isnull().mean()
    columns_to_drop = missing_ratios[missing_ratios > MISSING_VALUE_THRESHOLD].index.tolist()
    print(f"Additional columns to drop due to missing values: {columns_to_drop}")
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")
    
    # Impute missing values for numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    if IMPUTATION_METHOD == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif IMPUTATION_METHOD == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

# Function to process all partitioned Parquet files
def clean_partitioned_data(input_folder, output_folder, folder_pattern='partition_id=*'):
    # Find all Parquet files
    file_paths = glob(os.path.join(input_folder, f"{folder_pattern}/*.parquet"))
    print(f"Found {len(file_paths)} files in {input_folder} ({folder_pattern})")
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Clean the DataFrame
        df_cleaned = clean_dataframe(df)
        
        # Save the cleaned data
        relative_path = os.path.relpath(file_path, input_folder)
        output_file_path = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # Create subdirectories if needed
        df_cleaned.to_parquet(output_file_path, index=False)
        print(f"Cleaned file saved to: {output_file_path}")

# Run the cleaning process
if __name__ == "__main__":
    print("Cleaning Train Data...")
    clean_partitioned_data(
        input_folder="../data/train.parquet",
        output_folder="../data/cleaned/train"
    )
    
    print("Cleaning Test Data...")
    clean_partitioned_data(
        input_folder="../data/test.parquet",
        output_folder="../data/cleaned/test",
        folder_pattern="date_id=*"
    )
    
    print("Cleaning Lags Data...")
    clean_partitioned_data(
        input_folder="../data/lags.parquet",
        output_folder="../data/cleaned/lags",
        folder_pattern="date_id=*"
    )
