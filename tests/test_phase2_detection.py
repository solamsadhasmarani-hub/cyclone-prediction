from pathlib import Path

import torch

from ai.detection.dataset import generate_detection_demo_dataset
from ai.detection.model import build_detector_model, predict_image
from ai.detection.train import train_detector


def test_detection_dataset_generation(tmp_path: Path):
    metadata_path = generate_detection_demo_dataset(project_root=tmp_path, positive_events=2, negative_events=2, frames_per_event=2)
    assert metadata_path.exists()
    assert metadata_path.name == "detection_demo_metadata.csv"

    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + (2 * 2) + (2 * 2)

    model = build_detector_model(num_classes=2)
    assert isinstance(model, torch.nn.Module)


def test_detection_model_training_and_inference(tmp_path: Path):
    results = train_detector(project_root=tmp_path, epochs=1, batch_size=4)
    model_path = Path(results["model_path"])
    assert model_path.exists()

    positive_image = tmp_path / "data" / "raw" / "detection_demo" / "CYC_01_01.png"
    prediction = predict_image(positive_image, model_path)
    assert "cyclone_detected" in prediction
    assert prediction["cyclone_detected"] in {True, False}
    assert 0.0 <= float(prediction["confidence"]) <= 1.0
