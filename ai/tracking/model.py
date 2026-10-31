from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from ai.tracking.dataset import load_track_dataset

FEATURE_COLUMNS = ["latitude", "longitude", "hours_since_start"]
TARGET_COLUMNS = ["future_latitude", "future_longitude"]


def _train_validation_test_split(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by cyclone event to avoid leakage across track histories."""
    unique_events = dataframe["cyclone_id"].drop_duplicates().tolist()
    train_events = unique_events[: int(len(unique_events) * 0.7)]
    val_events = unique_events[int(len(unique_events) * 0.7): int(len(unique_events) * 0.85)]
    test_events = unique_events[int(len(unique_events) * 0.85):]

    train_frame = dataframe[dataframe["cyclone_id"].isin(train_events)].copy()
    validation_frame = dataframe[dataframe["cyclone_id"].isin(val_events)].copy()
    test_frame = dataframe[dataframe["cyclone_id"].isin(test_events)].copy()
    return train_frame, validation_frame, test_frame


def train_track_model(metadata_path: str | Path, model_path: str | Path | None = None) -> dict[str, object]:
    """Train a simple Random Forest baseline for future cyclone position forecasting."""
    dataframe = load_track_dataset(metadata_path)
    train_frame, _, test_frame = _train_validation_test_split(dataframe)

    X_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame[TARGET_COLUMNS]
    X_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame[TARGET_COLUMNS]

    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse_lat = np.sqrt(mean_squared_error(y_test["future_latitude"], predictions[:, 0]))
    rmse_lon = np.sqrt(mean_squared_error(y_test["future_longitude"], predictions[:, 1]))

    output_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parents[2] / "models" / "track_model.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, output_path)

    return {
        "model_path": str(output_path),
        "rmse_latitude": float(rmse_lat),
        "rmse_longitude": float(rmse_lon),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
    }


def evaluate_track_model(model_path: str | Path, metadata_path: str | Path) -> dict[str, object]:
    """Evaluate the model on the test set using RMSE for latitude and longitude."""
    import joblib

    dataframe = load_track_dataset(metadata_path)
    _, _, test_frame = _train_validation_test_split(dataframe)
    model = joblib.load(model_path)
    predictions = model.predict(test_frame[FEATURE_COLUMNS])
    true_values = test_frame[TARGET_COLUMNS].to_numpy()

    return {
        "rmse_latitude": float(np.sqrt(mean_squared_error(true_values[:, 0], predictions[:, 0]))),
        "rmse_longitude": float(np.sqrt(mean_squared_error(true_values[:, 1], predictions[:, 1]))),
        "test_rows": int(len(test_frame)),
    }


def predict_track(model_path: str | Path, feature_row: dict[str, float | int]) -> dict[str, object]:
    """Predict future cyclone position for a single record."""
    import joblib

    model = joblib.load(model_path)
    feature_frame = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    prediction = model.predict(feature_frame)[0]
    horizon = int(feature_row.get("prediction_horizon_hours", 12))
    return {
        "hours": horizon,
        "latitude": float(prediction[0]),
        "longitude": float(prediction[1]),
    }


if __name__ == "__main__":
    metadata_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "track_demo_metadata.csv"
    if not metadata_path.exists():
        from ai.tracking.dataset import generate_track_demo_dataset
        metadata_path = generate_track_demo_dataset()
    result = train_track_model(metadata_path)
    print(json.dumps(result, indent=2))
