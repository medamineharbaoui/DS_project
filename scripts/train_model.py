import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import pickle
import gc

# Configuration for features and target
class CONFIG:
    seed = 286
    target_col = "responder_6"
    feature_cols = [f"feature_{idx:02d}" for idx in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]

# Load training data
print("Loading training data...")
train = pl.scan_parquet("../mergeddata/training.parquet/").collect().to_pandas()

# Handle missing values in training data
X_train = train[CONFIG.feature_cols].fillna(method="ffill").fillna(0)
y_train = train[CONFIG.target_col]
w_train = train["weight"]

# Load validation data
print("Loading validation data...")
valid = pl.scan_parquet("../mergeddata/validation.parquet/").collect().to_pandas()

# Handle missing values in validation data
X_valid = valid[CONFIG.feature_cols].fillna(method="ffill").fillna(0)
y_valid = valid[CONFIG.target_col]
w_valid = valid["weight"]

# Free up memory
del train, valid
gc.collect()

# Define the XGBoost model
print("Initializing XGBoost model...")
xgb_model = XGBRegressor(
    n_estimators=1000,       # Number of boosting rounds
    learning_rate=0.05,      # Learning rate
    max_depth=6,             # Maximum depth of trees
    subsample=0.8,           # Subsample ratio of the training instances
    colsample_bytree=0.8,    # Subsample ratio of features
    random_state=CONFIG.seed # Seed for reproducibility
)

# Train the model
print("Training the model...")
xgb_model.fit(
    X_train,
    y_train,
    sample_weight=w_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    early_stopping_rounds=50,
    verbose=10  # Show training progress every 10 iterations
)

# Generate predictions for the validation dataset
print("Generating predictions...")
y_pred_valid_xgb = xgb_model.predict(X_valid)

# Calculate the R² score
valid_score = r2_score(y_valid, y_pred_valid_xgb, sample_weight=w_valid)
print(f"XGBoost Validation R² Score: {valid_score:.4f}")

# Save the trained model to a file
print("Saving the trained model...")
with open("../models/trained_xgb_model.pkl", "wb") as f:
    pickle.dump({"model": xgb_model}, f)

print("Model training and saving completed.")
