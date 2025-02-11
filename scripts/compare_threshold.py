import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Parameters
TEST_DATA_PATH = "../data/cleaned/test/date_id=0/part-0.parquet"
TRAIN_SELECTED_DATA_PATH = "../data/cleaned/train_selected/partition_id=0/part-0.parquet" # first file only to check
LAGS_DATA_PATH = "../data/cleaned/lags/date_id=0/part-0.parquet"
TARGET_COLUMN = "responder_6"
MODEL_PATHS = [
    "../models/model_threshold_0.05.pkl",
    "../models/model_threshold_0.1.pkl",
    "../models/model_threshold_0.2.pkl"
]

# Load test data
df_test = pd.read_parquet(TEST_DATA_PATH)
print("Columns in test dataset:", df_test.columns)

df_lags = pd.read_parquet(LAGS_DATA_PATH)
print("Columns in lags dataset:", df_test.columns)

df_train = pd.read_parquet(TRAIN_SELECTED_DATA_PATH)
print("Columns in train_selected dataset:", df_test.columns)

if TARGET_COLUMN not in df_test.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in test dataset.")

X_test = df_test.drop(columns=[TARGET_COLUMN])
y_test = df_test[TARGET_COLUMN]

# Evaluate each model
results = []
for model_path in MODEL_PATHS:
    model = joblib.load(model_path)
    print(f"Evaluating model: {model_path}")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_path,
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    })

results_df = pd.DataFrame(results)
print("\nComparison of Models:")
print(results_df)

results_df.to_csv("model_comparison_results.csv", index=False)
