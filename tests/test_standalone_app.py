from __future__ import annotations

from io import BytesIO

import pandas as pd
from PIL import Image
import pytest

from ai.config.imd_classes import IMD_CLASSES
from backend.api import app
from train_model import load_manifest, split_by_group
from utils.preprocessing import preprocess_image
from utils.temporal_analysis import (
    derive_evolution_status,
    timestamp_from_filename,
)


def test_image_resize_returns_rgb_224_square_and_rejects_corrupt_images():
    stream = BytesIO()
    Image.new("RGBA", (80, 50), color=(30, 80, 110, 255)).save(stream, format="PNG")
    array = preprocess_image(stream.getvalue())
    assert array.shape == (224, 224, 3)
    assert str(array.dtype) == "float32"
    with pytest.raises(ValueError, match="not a valid"):
        preprocess_image(b"not an image")


def test_temporal_trend_uses_timestamp_sorted_imd_class_sequence():
    observations = [
        {"timestamp": "2026-10-07T14:00:00Z", "predicted_class": "Severe Cyclonic Storm"},
        {"timestamp": "2026-10-07T13:00:00Z", "predicted_class": "Cyclonic Storm"},
    ]
    assert derive_evolution_status(observations) == "STRENGTHENING"
    observations[0]["predicted_class"] = "Cyclonic Storm"
    assert derive_evolution_status(observations) == "PEAK"
    observations[0]["predicted_class"] = "Depression"
    assert derive_evolution_status(observations) == "WEAKENING"
    assert derive_evolution_status(observations[:1]) == "INSUFFICIENT TEMPORAL DATA"


def test_timestamp_parser_avoids_treating_numeric_image_suffix_as_time():
    assert timestamp_from_filename("storm_20261007_02.png") == "2026-10-07T00:00:00Z"
    assert timestamp_from_filename("storm_20261007T141530.png") == "2026-10-07T14:15:30Z"
    assert timestamp_from_filename("storm_2026-10-07_14-15.png") == "2026-10-07T14:15:00Z"


def test_group_split_has_no_storm_or_event_leakage():
    rows = []
    for group_index in range(32):
        rows.append(
            {
                "group_id": f"event-{group_index}",
                "cyclone_detected": 1,
                "imd_class": IMD_CLASSES[group_index % len(IMD_CLASSES)],
            }
        )
        rows.append(
            {
                "group_id": f"event-{group_index}",
                "cyclone_detected": 0,
                "imd_class": pd.NA,
            }
        )
    splits = split_by_group(pd.DataFrame(rows))
    group_sets = {name: set(frame["group_id"]) for name, frame in splits.items()}
    assert not group_sets["train"] & group_sets["validation"]
    assert not group_sets["train"] & group_sets["test"]
    assert not group_sets["validation"] & group_sets["test"]
    assert all(set(frame["cyclone_detected"]) == {0, 1} for frame in splits.values())


def test_training_requires_real_manifest_instead_of_falling_back_to_demo_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="do not use demo data"):
        load_manifest(tmp_path / "missing.csv")


def test_flask_health_reports_prototype_model_provenance_and_upload_runs_checkpoint():
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json["model_ready"] is True
    assert health.json["detection_model_ready"] is True
    assert health.json["classification_model_ready"] is True
    assert health.json["intensity_model_ready"] is False
    assert health.json["track_model_ready"] is False
    assert "not been validated on real satellite imagery" in health.json["message"]

    upload = BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(upload, format="PNG")
    response = client.post(
        "/predict",
        data={"file": (BytesIO(upload.getvalue()), "image.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.json["data_mode"] == "SYNTHETIC-TRAINED PROTOTYPE INFERENCE"
    assert response.json["model_provenance"] == health.json["model_provenance"]
    assert isinstance(response.json["cyclone_detected"], bool)


def test_flask_requires_image_upload():
    response = app.test_client().post("/predict")
    assert response.status_code == 400
    assert "Upload one image" in response.json["error"]


def test_flask_rejects_corrupt_image_before_model_availability_check():
    response = app.test_client().post(
        "/predict",
        data={"file": (BytesIO(b"invalid image bytes"), "bad.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.json["status"] == "error"
    assert "not a valid" in response.json["error"]
