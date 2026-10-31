from pathlib import Path

import numpy as np

from ai.classification.dataset import generate_classification_demo_dataset
from ai.classification.model import build_classifier_model
from ai.explainability.gradcam import explain_prediction, generate_gradcam_heatmap


def test_gradcam_generation(tmp_path: Path):
    metadata_path = generate_classification_demo_dataset(project_root=tmp_path, events_per_class=2, frames_per_event=2)
    image_path = tmp_path / "data" / "raw" / "classification_demo" / "DEVELOPING_01_01.png"

    assert image_path.exists(), "Synthetic classification dataset should have been generated for explainability testing"

    model = build_classifier_model(num_classes=4)
    heatmap = generate_gradcam_heatmap(model, image_path)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

    explanation = explain_prediction(model, image_path)
    assert "predicted_class" in explanation
    assert "confidence" in explanation
    assert 0.0 <= float(explanation["confidence"]) <= 1.0
