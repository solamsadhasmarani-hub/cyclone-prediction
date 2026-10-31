from pathlib import Path

from ai.intensity.dataset import generate_intensity_demo_dataset
from ai.intensity.model import evaluate_intensity_model, predict_intensity, train_intensity_model


def test_intensity_demo_dataset_generation(tmp_path: Path):
    metadata_path = generate_intensity_demo_dataset(project_root=tmp_path, events=3)
    assert metadata_path.exists()
    assert metadata_path.name == "intensity_demo_metadata.csv"
    text = metadata_path.read_text(encoding="utf-8")
    assert "future_intensity" in text
    assert "current_intensity" in text


def test_intensity_training_and_prediction(tmp_path: Path):
    metadata_path = generate_intensity_demo_dataset(project_root=tmp_path, events=3)
    results = train_intensity_model(metadata_path, model_path=tmp_path / "models" / "intensity_model.joblib")

    assert results["mae"] >= 0.0
    assert results["rmse"] >= 0.0
    assert (tmp_path / "models" / "intensity_model.joblib").exists()

    sample_row = {
        "current_intensity": 40,
        "historical_intensity": 34,
        "wind_speed": 42,
        "pressure": 1000,
        "time_index_hours": 12,
        "latitude": 15.0,
        "longitude": 83.0,
        "prediction_horizon_hours": 12,
    }
    prediction = predict_intensity(tmp_path / "models" / "intensity_model.joblib", sample_row)
    assert "current_intensity" in prediction
    assert "predicted_intensity" in prediction
    assert "prediction_horizon_hours" in prediction
    assert prediction["prediction_horizon_hours"] == 12

    eval_summary = evaluate_intensity_model(tmp_path / "models" / "intensity_model.joblib", metadata_path)
    assert "mae" in eval_summary
    assert "rmse" in eval_summary
