from pathlib import Path

import torch

from ai.classification.dataset import CLASS_NAMES, generate_classification_demo_dataset
from ai.classification.model import build_classifier_model, predict_image
from ai.classification.train import evaluate_classifier, train_classifier


def test_classification_dataset_generation(tmp_path: Path):
    metadata_path = generate_classification_demo_dataset(project_root=tmp_path, events_per_class=3, frames_per_event=2)
    assert metadata_path.exists()
    assert metadata_path.name == "classification_demo_metadata.csv"

    dataset_text = metadata_path.read_text(encoding="utf-8")
    assert "class_name" in dataset_text
    assert "label" in dataset_text

    model = build_classifier_model(num_classes=4)
    assert isinstance(model, torch.nn.Module)


def test_classification_training_and_inference(tmp_path: Path):
    results = train_classifier(project_root=tmp_path, epochs=1, batch_size=4)
    model_path = Path(results["model_path"])
    assert model_path.exists()

    assert "evaluation" in results
    assert "accuracy" in results["evaluation"]

    sample_image = tmp_path / "data" / "raw" / "classification_demo" / "DEVELOPING_01_01.png"
    prediction = predict_image(sample_image, model_path, class_names=CLASS_NAMES)
    assert prediction["classification"] in CLASS_NAMES
    assert 0.0 <= float(prediction["confidence"]) <= 1.0
