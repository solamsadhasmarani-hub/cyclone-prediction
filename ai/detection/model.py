from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18


def build_detector_model(num_classes: int = 2) -> nn.Module:
    """Create a lightweight ResNet-18 detector that outputs cyclone/no-cyclone probabilities."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_detector_model(model_path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Load a saved detector model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model = build_detector_model(num_classes=checkpoint.get("num_classes", 2))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _prediction_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_image(image_path: str | Path, model_path: str | Path, device: str | torch.device = "cpu") -> dict[str, float | bool | str]:
    """Run a single inference pass and return the detection decision and confidence."""
    model = load_detector_model(model_path, device=device)
    image = Image.open(image_path).convert("RGB")
    tensor = _prediction_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    cyclone_prob = float(probabilities[1].item())
    no_cyclone_prob = float(probabilities[0].item())
    cyclone_detected = cyclone_prob >= 0.5

    return {
        "cyclone_detected": cyclone_detected,
        "confidence": cyclone_prob if cyclone_detected else no_cyclone_prob,
        "detection_confidence": cyclone_prob if cyclone_detected else no_cyclone_prob,
        "class_probabilities": {"no_cyclone": no_cyclone_prob, "cyclone": cyclone_prob},
        "label": "cyclone" if cyclone_detected else "no_cyclone",
    }


if __name__ == "__main__":
    example = predict_image("data/raw/detection_demo/CYC_01_01.png", "models/cyclone_detector_resnet18.pt")
    print(json.dumps(example, indent=2))
