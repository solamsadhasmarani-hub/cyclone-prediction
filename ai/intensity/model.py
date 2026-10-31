from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from ai.intensity.dataset import load_intensity_dataset

FEATURE_COLUMNS = [
    "current_intensity",
    "historical_intensity",
    "wind_speed",
    "pressure",
    "time_index_hours",
    "latitude",
    "longitude",
]


def _train_validation_test_split(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a leakage-safe split across cyclone events."""
    unique_events = dataframe["cyclone_id"].drop_duplicates().tolist()
    train_events = unique_events[: int(len(unique_events) * 0.7)]
    val_events = unique_events[int(len(unique_events) * 0.7): int(len(unique_events) * 0.85)]
    test_events = unique_events[int(len(unique_events) * 0.85):]

    train_frame = dataframe[dataframe["cyclone_id"].isin(train_events)].copy()
    validation_frame = dataframe[dataframe["cyclone_id"].isin(val_events)].copy()
    test_frame = dataframe[dataframe["cyclone_id"].isin(test_events)].copy()
    return train_frame, validation_frame, test_frame


def train_intensity_model(metadata_path: str | Path, model_path: str | Path | None = None) -> dict[str, object]:
    """Train a Random Forest baseline for intensity prediction."""
    dataframe = load_intensity_dataset(metadata_path)
    train_frame, validation_frame, test_frame = _train_validation_test_split(dataframe)

    X_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["future_intensity"]
    X_val = validation_frame[FEATURE_COLUMNS]
    y_val = validation_frame["future_intensity"]
    X_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["future_intensity"]

    model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    root = Path(model_path) if model_path is not None else Path(__file__).resolve().parents[2] / "models" / "intensity_model.joblib"
    root.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, root)

    return {
        "model_path": str(root),
        "mae": float(mae),
        "rmse": float(rmse),
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
    }


def evaluate_intensity_model(model_path: str | Path, metadata_path: str | Path) -> dict[str, object]:
    """Evaluate the trained model on the test split using MAE and RMSE."""
    import joblib

    dataframe = load_intensity_dataset(metadata_path)
    _, _, test_frame = _train_validation_test_split(dataframe)
    model = joblib.load(model_path)
    predictions = model.predict(test_frame[FEATURE_COLUMNS])
    true_values = test_frame["future_intensity"].to_numpy()

    return {
        "mae": float(mean_absolute_error(true_values, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(true_values, predictions))),
        "test_rows": int(len(test_frame)),
    }


def predict_intensity(model_path: str | Path, feature_row: dict[str, float | int]) -> dict[str, object]:
    """Predict the next-horizon intensity for a single feature record."""
    import joblib

    model = joblib.load(model_path)
    feature_frame = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    prediction = float(model.predict(feature_frame)[0])
    current_intensity = float(feature_row.get("current_intensity", 0.0))
    horizon = int(feature_row.get("prediction_horizon_hours", 12))

    return {
        "current_intensity": current_intensity,
        "predicted_intensity": prediction,
        "prediction_horizon_hours": horizon,
    }


if __name__ == "__main__":
    metadata_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "intensity_demo_metadata.csv"
    if not metadata_path.exists():
        from ai.intensity.dataset import generate_intensity_demo_dataset
        metadata_path = generate_intensity_demo_dataset()
    result = train_intensity_model(metadata_path)
    print(json.dumps(result, indent=2))
