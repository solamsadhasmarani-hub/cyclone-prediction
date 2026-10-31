from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ai.classification.dataset import CLASS_NAMES
from ai.classification.model import load_classifier_model
from ai.detection.model import load_detector_model
from ai.explainability.gradcam import GradCAM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATH = PROJECT_ROOT / "models" / "cyclone_detector_resnet18.pt"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "cyclone_classifier_resnet18.pt"
MODEL_ARCHITECTURE = "ResNet-18 detector and four-stage classifier"
MODEL_PROVENANCE = (
    "These prototype checkpoints were trained on generated synthetic demo images "
    "and have not been validated on real satellite imagery."
)
DATA_MODE = "SYNTHETIC-TRAINED PROTOTYPE INFERENCE"


class PrototypeInferenceUnavailableError(RuntimeError):
    """Raised when an existing prototype checkpoint cannot serve inference."""


@lru_cache(maxsize=1)
def _prediction_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@lru_cache(maxsize=1)
def _load_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    missing = [
        path.name for path in (DETECTOR_PATH, CLASSIFIER_PATH)
        if not path.is_file()
    ]
    if missing:
        raise PrototypeInferenceUnavailableError(
            "Prototype inference checkpoints are missing: " + ", ".join(missing)
        )
    try:
        return (
            load_detector_model(DETECTOR_PATH),
            load_classifier_model(CLASSIFIER_PATH),
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        raise PrototypeInferenceUnavailableError(
            f"Could not load the ResNet-18 prototype checkpoints: {error}"
        ) from error


def get_model_status() -> dict[str, Any]:
    detector_available = DETECTOR_PATH.is_file()
    classifier_available = CLASSIFIER_PATH.is_file()
    return {
        "detection_model_ready": detector_available,
        "classification_model_ready": classifier_available,
        "intensity_model_ready": False,
        "track_model_ready": False,
        "gradcam_ready": classifier_available,
        "risk_model_ready": False,
        "prototype_only": True,
        "demo_mode": True,
        "model_architecture": MODEL_ARCHITECTURE,
        "model_provenance": MODEL_PROVENANCE,
    }


def _encode_explanation(image: Image.Image, heatmap: np.ndarray) -> str:
    grayscale = Image.fromarray(
        np.uint8(np.clip(heatmap, 0.0, 1.0) * 255),
        mode="L",
    ).resize(image.size, Image.Resampling.BILINEAR)
    mask = grayscale.point(lambda value: round(value * 0.55))
    red_overlay = Image.new("RGB", image.size, (255, 48, 24))
    explanation = Image.composite(red_overlay, image, mask)
    output = BytesIO()
    explanation.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def predict_uploaded_image(image_bytes: bytes) -> dict[str, Any]:
    """Run cached prototype detection, stage classification, and optional Grad-CAM."""
    detector, classifier = _load_models()
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError("The uploaded file is not a valid or readable image.") from error

    tensor = _prediction_transform()(image).unsqueeze(0)
    with torch.inference_mode():
        detection_probabilities = torch.softmax(detector(tensor), dim=1)[0]
    detected = int(detection_probabilities.argmax().item()) == 1
    result: dict[str, Any] = {
        "cyclone_detected": detected,
        "detection_confidence": float(detection_probabilities[1 if detected else 0].item()),
        "classification": None,
        "classification_confidence": None,
        "stage_probabilities": None,
        "imd_classification": None,
        "imd_confidence": None,
        "imd_probabilities": None,
        "explanation_image": None,
        "analysis_timestamp": None,
        "model_architecture": MODEL_ARCHITECTURE,
        "model_provenance": MODEL_PROVENANCE,
        "data_mode": DATA_MODE,
        "status": "success",
    }
    if not detected:
        return result

    with torch.inference_mode():
        class_probabilities = torch.softmax(classifier(tensor), dim=1)[0]
    class_index = int(class_probabilities.argmax().item())
    stage_probabilities = {
        stage: float(class_probabilities[index].item())
        for index, stage in enumerate(CLASS_NAMES)
    }
    result.update(
        {
            "classification": CLASS_NAMES[class_index],
            "classification_confidence": stage_probabilities[CLASS_NAMES[class_index]],
            "stage_probabilities": stage_probabilities,
            "explanation_image": _encode_explanation(
                image,
                GradCAM(classifier).generate(tensor, class_index=class_index),
            ),
        }
    )
    return result
