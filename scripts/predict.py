import numpy as np
import pandas as pd
import polars as pl
import pickle

# Configuration for features
class CONFIG:
    feature_cols = [f"feature_{idx:02d}" for idx in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]

# Load test data
test = pl.scan_parquet("../mergeddata/testing.parquet").collect().to_pandas()
print("Test data columns:", test.columns)

# Take only the first 39 rows (symbol_id from 0 to 38)
#test = test.head(39)

# Preprocess test data (handle missing values)
X_test = test[CONFIG.feature_cols].ffill().fillna(0)

# Load the saved model
with open("../models/trained_xgb_model.pkl", "rb") as f:
    result = pickle.load(f)
    xgb_model = result["model"]

# Make predictions
y_pred_test = xgb_model.predict(X_test)

# Clip predictions to standard range (-5, 5)
y_pred_test = np.clip(y_pred_test, a_min=-5, a_max=5)

# Format predictions for submission
predictions = pd.DataFrame({
    "row_id": test["id"],
    "responder_6": y_pred_test
})

# Save predictions
predictions.to_csv("../outputs/submission.csv", index=False)
print("Predictions saved to submission.csv!")
