import pandas as pd
import os
from glob import glob

def process_partitioned_parquet(folder_path, folder_pattern='partition_id=*', process_func=None):
    # Find all parquet files based on the pattern
    file_paths = glob(os.path.join(folder_path, f'{folder_pattern}/*.parquet'))
    print(f"Found {len(file_paths)} files in {folder_path} ({folder_pattern})")
    
    # Process each file
    for idx, file_path in enumerate(file_paths):  # Added index to handle file order
        print(f"Processing data from: {file_path}")
        df = pd.read_parquet(file_path)
        if process_func:
            process_func(df, file_path, idx)  # Pass data and index to processing function

# Function to filter numeric data, log removed columns, and analyze correlations
def analyze_data(df, file_path, idx):
    report = []
    
    # Use partition_id (or file path) to generate a unique file name
    partition_id = os.path.basename(file_path).replace('.parquet', '')  # Use partition ID as unique identifier
    file_name = f"exploring_report_{partition_id}_{idx}.txt"  # Added index for uniqueness

    # Filter numeric data and log removed columns
    non_numeric_columns = df.select_dtypes(exclude='number').columns
    non_numeric_count = len(non_numeric_columns)
    total_columns = len(df.columns)
    percentage_removed = (non_numeric_count / total_columns) * 100
    
    report.append("Removed Non-Numeric Columns:")
    report.append("="*50)
    report.append(f"Total Columns: {total_columns}")
    report.append(f"Non-Numeric Columns Removed: {non_numeric_count} ({percentage_removed:.2f}%)")
    report.append("\n".join(non_numeric_columns))

    # Filter out non-numeric columns for analysis
    numeric_df = df.select_dtypes(include='number')

    # Identify missing values
    missing_data = numeric_df.isnull().sum()
    total_rows = len(numeric_df)
    missing_percentage = (missing_data / total_rows) * 100

    report.append("\nMissing Data Analysis:")
    report.append("="*50)
    report.append("Column Name\tMissing Count\tPercentage")
    for column, count, percentage in zip(missing_data.index, missing_data, missing_percentage):
        report.append(f"{column}\t{count}\t{percentage:.2f}%")

    # Correlation Analysis
    report.append("\nCorrelation Analysis:")
    report.append("="*50)
    correlation_matrix = numeric_df.corr()
    report.append(str(correlation_matrix))

    # Write the report to a file and print to terminal
    with open(file_name, 'w') as f:
        for line in report:
            print(line)
            f.write(line + "\n")

# Main logic: Preview and analyze data incrementally
print("Processing and Analyzing Train Data:")
process_partitioned_parquet('../data/train.parquet', folder_pattern='partition_id=*', process_func=analyze_data)
