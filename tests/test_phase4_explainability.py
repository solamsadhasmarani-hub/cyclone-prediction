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
    target_layer = model.layer4
    forward_hook_count = len(target_layer._forward_hooks)
    backward_hook_count = len(target_layer._backward_hooks)
    heatmap = generate_gradcam_heatmap(model, image_path)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    assert len(target_layer._forward_hooks) == forward_hook_count
    assert len(target_layer._backward_hooks) == backward_hook_count

    explanation = explain_prediction(model, image_path)
    assert "predicted_class" in explanation
    assert "confidence" in explanation
    assert 0.0 <= float(explanation["confidence"]) <= 1.0


def test_gradcam_does_not_create_a_synthetic_image_for_a_missing_path(tmp_path: Path):
    missing_image = tmp_path / "missing.png"
    model = build_classifier_model(num_classes=4)

    try:
        generate_gradcam_heatmap(model, missing_image)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Grad-CAM should reject a missing image instead of generating demo input.")
    assert not missing_image.exists()
