#1
import pandas as pd
import os
from glob import glob



#df0 = pd.read_parquet("../data/train.parquet/partition_id=0/part-0.parquet")
#print("part0 : ",df0.shape)

# Helper function to load and preview partitioned Parquet data
def preview_partitioned_parquet(folder_path, folder_pattern='partition_id=*', rows=5):
    # Adjust the glob pattern to match the specified folder structure
    file_paths = glob(os.path.join(folder_path, f'{folder_pattern}/*.parquet'))
    print(f"Found {len(file_paths)} files in {folder_path} ({folder_pattern})")
    
    preview_data = []
    for file_path in file_paths[:3]:  # Limit to preview first 3 files for brevity
        print(f"Loading preview from: {file_path}")
        df = pd.read_parquet(file_path)
        preview_data.append(df.head(rows))  # Get first few rows from each file
    
    return pd.concat(preview_data, ignore_index=True)  # Combine previews

# Preview the data
#print("Previewing Train Data:")
#train_preview = preview_partitioned_parquet('../data/train.parquet', folder_pattern='partition_id=*', rows=5)
#print(train_preview.shape)

#Show columns and their types for Train data
#print("\nTrain Data - Columns and Types:")
#print(train_preview.dtypes)

#print("\nPreviewing Lags Data:")
#lags_preview = preview_partitioned_parquet('../data/lags.parquet', folder_pattern='date_id=*', rows=5)
#print(lags_preview)

# Show columns and their types for Lags data
#print("\nLags Data - Columns and Types:")
#print(lags_preview.dtypes)

#print("\nPreviewing Features Data:")
#features_preview = pd.read_csv('../data/features.csv', nrows=5)
#print(features_preview)

# Show columns and their types for Features data
#print("\nFeatures Data - Columns and Types:")
#print(features_preview.dtypes)

#print("\nPreviewing Responders Data:")
#responders_preview = pd.read_csv('../data/responders.csv', nrows=5)
#print(responders_preview)

# Show columns and their types for Responders data
#print("\nResponders Data - Columns and Types:")
#print(responders_preview.dtypes)


print("\nPreviewing Test Data:")
test_preview = preview_partitioned_parquet('../data/test.parquet', folder_pattern='date_id=*', rows=5)
print(test_preview)

# Show columns and their types for Test data
print("\nTest Data - Columns and Types:")
print(test_preview.dtypes)