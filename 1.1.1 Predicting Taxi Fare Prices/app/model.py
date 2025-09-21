"""
app/model.py
Model loader and prediction helpers for NYC Taxi Fare Prediction.
"""

import os
import json
import logging
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Default paths (can be overridden via environment variables)
# ---------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "taxi_fare_model.joblib")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "feature_columns.json")

# ---------------------------------------------------------------------
# Cached objects
# ---------------------------------------------------------------------
_model = None
_feature_columns: List[str] = []

# ---------------------------------------------------------------------
# Load model + feature columns
# ---------------------------------------------------------------------
def load_model(model_path: str = None, features_path: str = None):
    global _model, _feature_columns

    model_file = model_path or MODEL_PATH
    features_file = features_path or FEATURES_PATH

    # Load model if not already in memory
    if _model is None:
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}")
        logger.info(f"Loading model from {model_file}")
        loaded = joblib.load(model_file)
        if hasattr(loaded, "predict"):
            _model = loaded
        else:
            raise TypeError(f"Loaded object from {model_file} is not a model.")

    # Load feature columns if not already in memory
    if not _feature_columns:
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Feature columns file not found at {features_file}")
        logger.info(f"Loading feature columns from {features_file}")
        with open(features_file, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "features" in data:
            _feature_columns = data["features"]
        elif isinstance(data, list):
            _feature_columns = data
        else:
            raise ValueError(f"Unexpected content in {features_file}: {type(data)}")

    return _model, _feature_columns

# ---------------------------------------------------------------------
# Compute derived features
# ---------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute model features from raw input columns:
    - distance_miles from lat/lon
    - hour_of_day, day_of_week, month from pickup_datetime
    """
    df = df.copy()

    # Haversine distance
    def haversine(lat1, lon1, lat2, lon2):
        R = 3959.87433  # miles
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df["distance_miles"] = haversine(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.weekday
    df["month"] = df["pickup_datetime"].dt.month

    return df

# ---------------------------------------------------------------------
# Prepare dataframe for prediction
# ---------------------------------------------------------------------
def prepare_dataframe(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Ensure the DataFrame has exactly the model’s expected columns.
    - Missing columns are added with NaN.
    - Extra columns are ignored.
    """
    X = compute_features(df)
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0  # Fill missing numeric features with 0
    X = X[feature_columns]
    return X

# ---------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------
def predict_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the taxi fare model on a pandas DataFrame.
    Returns only the predicted fare in USD.
    """
    model, feature_columns = load_model()
    X = prepare_dataframe(df, feature_columns)

    try:
        preds = model.predict(X)
    except Exception:
        logger.exception("Error during prediction")
        raise

    results = [{"fare_usd": round(float(p), 2)} for p in preds]
    return {"results": results}
