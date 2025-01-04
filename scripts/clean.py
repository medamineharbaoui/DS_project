import pandas as pd
import os
import pyarrow.parquet as pq

# Define file paths
train_dir = "../data/preprocessed/train"
test_dir = "../data/preprocessed/test"
lags_dir = "../data/preprocessed/lags"
features_file = "../data/preprocessed/features.parquet"
responders_file = "../data/preprocessed/responders.parquet"
output_dir = "../data/processed"
os.makedirs(output_dir, exist_ok=True)

# Function to read Parquet in chunks using PyArrow
def load_parquet_in_chunks(file_path, batch_size=10000):
    table = pq.read_table(file_path)
    total_rows = table.num_rows
    for start in range(0, total_rows, batch_size):
        yield table.slice(start, batch_size).to_pandas()

# Function to load all data in a directory in chunks
def load_data_in_chunks(directory, batch_size=10000, file_type="parquet"):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_type):
                file_path = os.path.join(root, file)
                print(f"Loading file: {file_path}")
                for chunk in load_parquet_in_chunks(file_path, batch_size=batch_size):
                    yield chunk

# Data cleaning
def clean_data(df):
    num_cols = df.select_dtypes(include=["float32", "int16", "int8"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

# Merge lags in chunks
def merge_lags_in_chunks(data_gen, lags, batch_size=10000):
    for chunk in data_gen:
        chunk = clean_data(chunk)
        chunk = pd.merge(chunk, lags, on=["date_id", "time_id", "symbol_id"], how="left")
        yield chunk

# Save chunks to file
def save_to_parquet(data_gen, output_file):
    first_write = True
    for chunk in data_gen:
        chunk.to_parquet(output_file, mode="a", index=False, header=first_write)
        first_write = False

# Process lags data (entirely loaded as it is assumed smaller than train/test)
print("Loading lags data...")
lags_data = pd.concat(load_data_in_chunks(lags_dir), ignore_index=True)
lags_data = clean_data(lags_data)

# Process train data
print("Processing train data...")
train_gen = load_data_in_chunks(train_dir)
train_merged_gen = merge_lags_in_chunks(train_gen, lags_data)
save_to_parquet(train_merged_gen, os.path.join(output_dir, "train_processed.parquet"))

# Process test data
print("Processing test data...")
test_gen = load_data_in_chunks(test_dir)
test_merged_gen = merge_lags_in_chunks(test_gen, lags_data)
save_to_parquet(test_merged_gen, os.path.join(output_dir, "test_processed.parquet"))

# Process features and responders
print("Processing features and responders...")
features_data = pd.read_parquet(features_file)
responders_data = pd.read_parquet(responders_file)

def map_features_to_tags(features_df):
    features_df = features_df.set_index("feature")
    tag_cols = [col for col in features_df.columns if col.startswith("tag")]
    return features_df[tag_cols]

def map_responders_to_tags(responders_df):
    responders_df = responders_df.set_index("responder")
    tag_cols = [col for col in responders_df.columns if col.startswith("tag")]
    return responders_df[tag_cols]

feature_tags = map_features_to_tags(features_data)
responder_tags = map_responders_to_tags(responders_data)

# Save lags data separately
lags_data.to_parquet(os.path.join(output_dir, "lags_processed.parquet"))

print("Data processing complete. Files saved in:", output_dir)
