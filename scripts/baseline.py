import pandas as pd
import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
import dask.dataframe as dd


# Helper function to load Parquet files in smaller batches using Dask
def load_parquet_in_batches(directory, folder_pattern='partition_id=*', batch_size=500):
    file_paths = glob(os.path.join(directory, f'{folder_pattern}/*.parquet'))
    print(f"Found {len(file_paths)} files in {directory}")

    for file_path in file_paths:
        print(f"Loading file: {file_path}")
        
        # Read the Parquet file using Dask (reads lazily)
        df = dd.read_parquet(file_path)

        # Process the DataFrame in smaller batches
        # Dask DataFrames are already partitioned, we can convert them into chunks
        for i, chunk in enumerate(df.to_delayed()):
            # Convert each Dask chunk to a Pandas DataFrame when necessary
            yield chunk.compute()



# Function to process features in chunks and save to disk
def process_features_in_chunks(data_iterator, imputer=None, scaler=None, output_directory='processed_data'):
    # Initialize imputer and scaler if not provided
    imputer = imputer or SimpleImputer(strategy='median')
    scaler = scaler or StandardScaler()

    # Make sure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    chunk_counter = 0  # Track how many chunks we've processed
    for chunk in data_iterator:
        print(f"Processing chunk {chunk_counter + 1}")
        
        # Identify numeric columns
        num_cols = chunk.select_dtypes(include=["float32", "int16", "int8", "float64", "int32"]).columns
        
        # Drop columns with all NaNs
        non_missing_cols = chunk[num_cols].columns[chunk[num_cols].notna().any()]
        missing_cols = set(num_cols) - set(non_missing_cols)
        if missing_cols:
            print(f"Skipping columns with all NaN values: {missing_cols}")
        
        # Select valid columns for processing
        valid_chunk = chunk[non_missing_cols]

        # Impute missing values in the valid columns
        imputed_data = imputer.fit_transform(valid_chunk)
        
        # Scale the imputed data
        scaled_data = scaler.fit_transform(imputed_data)

        # Create a DataFrame for the processed data
        processed_chunk = pd.DataFrame(scaled_data, columns=valid_chunk.columns, index=valid_chunk.index)

        # Merge back non-numeric columns (unchanged columns)
        non_numeric_cols = chunk.drop(columns=num_cols)
        processed_chunk = pd.concat([non_numeric_cols, processed_chunk], axis=1)

        # Save the processed chunk to disk
        processed_chunk.to_parquet(os.path.join(output_directory, f"processed_chunk_{chunk_counter}.parquet"))

        chunk_counter += 1

    print(f"All chunks processed and saved to {output_directory}.")



# Incremental training for models
def train_model_incrementally(data_iterator, chunk_size=500):
    lgb_model = lgb.LGBMRegressor()
    xgb_model = xgb.XGBRegressor()
    
    for i, chunk in enumerate(data_iterator):
        print(f"Processing chunk {i + 1}")
        if 'target' not in chunk:
            raise ValueError("The 'target' column is missing from the data.")
        
        X_chunk = chunk.drop(columns=['target'])
        y_chunk = chunk['target']
        
        X_chunk_train, X_chunk_val, y_chunk_train, y_chunk_val = train_test_split(
            X_chunk, y_chunk, test_size=0.2, random_state=42
        )
        
        # Incremental training for LightGBM
        lgb_model.fit(X_chunk_train, y_chunk_train, init_model=lgb_model if i > 0 else None)
        lgb_preds = lgb_model.predict(X_chunk_val)
        lgb_r2 = r2_score(y_chunk_val, lgb_preds)
        
        # Incremental training for XGBoost
        xgb_model.fit(X_chunk_train, y_chunk_train, xgb_model if i > 0 else None)
        xgb_preds = xgb_model.predict(X_chunk_val)
        xgb_r2 = r2_score(y_chunk_val, xgb_preds)
        
        print(f"Chunk {i + 1}: LightGBM R²: {lgb_r2}, XGBoost R²: {xgb_r2}")
    
    return lgb_model, xgb_model

# Load, process, and train incrementally
train_iterator = load_parquet_in_batches('../data/preprocessed/train', folder_pattern='partition_id=*', batch_size=500)
processed_iterator = process_features_in_chunks(train_iterator)

lgb_model, xgb_model = train_model_incrementally(processed_iterator)

# Save the trained models
import joblib
joblib.dump(lgb_model, 'final_lgb_model.pkl')
joblib.dump(xgb_model, 'final_xgb_model.pkl')


# understand the data
# choosing a model to work with
# preprocessing
# there is a correlation between features - I should use it to remove some feature
# make a first submission to kaggle