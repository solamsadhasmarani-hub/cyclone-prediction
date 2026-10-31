from pathlib import Path

from ai.tracking.dataset import generate_track_demo_dataset
from ai.tracking.model import evaluate_track_model, predict_track, train_track_model


def test_track_demo_dataset_generation(tmp_path: Path):
    metadata_path = generate_track_demo_dataset(project_root=tmp_path, events=3)
    assert metadata_path.exists()
    assert metadata_path.name == "track_demo_metadata.csv"
    text = metadata_path.read_text(encoding="utf-8")
    assert "future_latitude" in text
    assert "future_longitude" in text


def test_track_training_and_prediction(tmp_path: Path):
    metadata_path = generate_track_demo_dataset(project_root=tmp_path, events=3)
    results = train_track_model(metadata_path, model_path=tmp_path / "models" / "track_model.joblib")

    assert "rmse_latitude" in results
    assert "rmse_longitude" in results
    assert (tmp_path / "models" / "track_model.joblib").exists()

    sample_row = {
        "latitude": 12.0,
        "longitude": 82.0,
        "hours_since_start": 12,
        "prediction_horizon_hours": 12,
    }
    prediction = predict_track(tmp_path / "models" / "track_model.joblib", sample_row)
    assert "hours" in prediction
    assert "latitude" in prediction
    assert "longitude" in prediction

    eval_summary = evaluate_track_model(tmp_path / "models" / "track_model.joblib", metadata_path)
    assert "rmse_latitude" in eval_summary
    assert "rmse_longitude" in eval_summary
