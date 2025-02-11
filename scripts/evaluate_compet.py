import pandas as pd
import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error

# Load predictions
predictions = pd.read_csv("../outputs/submission.csv")

# Load ground truth data for date_id 1697 and 1698
true_data_1697 = pl.scan_parquet("../mergeddata/testing.parquet/date_id=1697").collect().to_pandas()
true_data_1698 = pl.scan_parquet("../mergeddata/testing.parquet/date_id=1698").collect().to_pandas()

# Combine the ground truth data
true_data = pd.concat([true_data_1697, true_data_1698], ignore_index=True)


# print("Predictions Columns:", predictions.columns)
# print("True Data Columns:", true_data.columns)


# Merge predictions with ground truth based on row_id
merged = predictions.merge(true_data, left_on="row_id", right_on="id", suffixes=("_pred", "_true"))


# Extract true and predicted values
y_true = merged["responder_6_true"]
y_pred = merged["responder_6_pred"]

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Compute MASE (Mean Absolute Scaled Error)
# First, compute naive forecast as the previous value (shifting by one)
naive_forecast = y_true.shift(1).dropna()
actual_differences = np.abs(np.diff(y_true))

# MASE formula: MAE / mean(abs(naive_forecast_error))
mae = np.mean(np.abs(y_pred - y_true))
mase = mae / np.mean(actual_differences)

# Print results
print(f"📊 RMSE: {rmse:.4f}")
print(f"📊 MASE: {mase:.4f}")
