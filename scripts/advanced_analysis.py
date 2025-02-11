#2
import pandas as pd
import os
import json
from glob import glob

# Columns to exclude from correlation analysis
excluded_columns = ["date_id", "time_id", "symbol_id", "weight"]

# Initialize a dictionary to store aggregated results
aggregate_data = {
    "summary": {
        "total_files": 0,
        "total_rows": 0,
        "total_columns": 0,
    },
    "missing_data": {},  # To store missing data counts and percentages
    "correlations": {},  # To store the global correlation matrix
    "excluded_columns": excluded_columns,  # List of excluded columns
}

def process_partitioned_parquet(folder_path, folder_pattern='partition_id=*', process_func=None):
    # Find all parquet files based on the pattern
    file_paths = glob(os.path.join(folder_path, f'{folder_pattern}/*.parquet'))
    print(f"Found {len(file_paths)} files in {folder_path} ({folder_pattern})")
    
    # Process each file
    for idx, file_path in enumerate(file_paths):
        print(f"Processing data from: {file_path}")
        df = pd.read_parquet(file_path)
        if process_func:
            process_func(df, file_path, idx)

def analyze_data(df, file_path, idx):
    global aggregate_data
    
    # Update summary
    aggregate_data["summary"]["total_files"] += 1
    aggregate_data["summary"]["total_rows"] += len(df)
    aggregate_data["summary"]["total_columns"] = max(aggregate_data["summary"]["total_columns"], len(df.columns))

    # Missing Data Analysis
    missing_data = df.isnull().sum()
    total_rows = len(df)
    missing_percentage = (missing_data / total_rows) * 100

    total_missing_count = 0  # Total missing count across all columns
    total_missing_percentage = 0.0  # Total missing percentage across all columns

    for column, count, percentage in zip(missing_data.index, missing_data, missing_percentage):
        if column not in aggregate_data["missing_data"]:
            aggregate_data["missing_data"][column] = {"missing_count": 0, "missing_percentage": 0.0}
        aggregate_data["missing_data"][column]["missing_count"] += count
        aggregate_data["missing_data"][column]["missing_percentage"] += percentage / aggregate_data["summary"]["total_files"]

        # Aggregate total missing counts and percentages
        total_missing_count += count
        total_missing_percentage += percentage

    # Add total missing counts and overall missing percentage to the summary
    aggregate_data["summary"]["total_missing_count"] = total_missing_count
    aggregate_data["summary"]["overall_missing_percentage"] = total_missing_percentage / aggregate_data["summary"]["total_files"]

    # Excluded columns (shouldn't be included in correlation analysis)
    excluded_columns = ["date_id", "time_id", "symbol_id", "weight"]

    # Correlation Analysis for numeric columns
    numeric_df = df.select_dtypes(include='number')

    # Exclude the columns listed in excluded_columns from the correlation analysis
    numeric_df = numeric_df.drop(columns=[col for col in excluded_columns if col in numeric_df.columns])

    correlation_matrix = numeric_df.corr()

    # Loop through the correlation matrix and exclude diagonal elements (correlation of a feature with itself)
    for col1 in correlation_matrix.columns:
        if col1 not in aggregate_data["correlations"]:
            aggregate_data["correlations"][col1] = {}
        for col2 in correlation_matrix.columns:
            if col1 != col2:  # Exclude correlation with itself
                if col2 not in aggregate_data["correlations"][col1]:
                    aggregate_data["correlations"][col1][col2] = 0.0
                aggregate_data["correlations"][col1][col2] += correlation_matrix.at[col1, col2] / aggregate_data["summary"]["total_files"]



# Finalize missing data percentages after processing all files
def finalize_missing_data():
    total_rows = aggregate_data["summary"]["total_rows"]
    for column, stats in aggregate_data["missing_data"].items():
        stats["missing_percentage"] = (stats["missing_count"] / total_rows) * 100

# Process the dataset and aggregate results
print("Processing and Analyzing Train Data:")
process_partitioned_parquet('../data/train.parquet', folder_pattern='partition_id=*', process_func=analyze_data)

# Finalize aggregated data
finalize_missing_data()

# Save the aggregated results to a JSON file
output_json_file = "aggregated_analysis.json"
with open(output_json_file, 'w') as f:
    json.dump(aggregate_data, f, indent=4)

print(f"\nAnalysis aggregated and saved to {output_json_file}")


# features with no correlation NAN : 
# 00, 01, 02, 03, 04, 21, 26, 27, 31 - we can remove these !