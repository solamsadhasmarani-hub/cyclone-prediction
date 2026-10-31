from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from ai.classification.dataset import CLASS_NAMES


def build_classifier_model(num_classes: int = 4) -> nn.Module:
    """Create a transfer-learning-style ResNet-18 model for multi-class cyclone stage classification."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_classifier_model(model_path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Load a saved classifier checkpoint."""
    return _load_classifier_model_cached(str(Path(model_path).resolve()), str(torch.device(device)))


@lru_cache(maxsize=4)
def _load_classifier_model_cached(model_path: str, device: str) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    class_names = checkpoint.get("class_names")
    if class_names is not None and class_names != CLASS_NAMES:
        raise ValueError("Classifier checkpoint labels do not match the supported prototype stages.")
    model = build_classifier_model(num_classes=checkpoint.get("num_classes", 4))
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


def predict_image(image_path: str | Path, model_path: str | Path, class_names: list[str] | None = None, device: str | torch.device = "cpu") -> dict[str, object]:
    """Run a single-stage inference pass and return the predicted class and confidence."""
    if class_names is None:
        class_names = ["developing", "mature", "weakening", "dissipating"]

    model = load_classifier_model(model_path, device=device)
    image = Image.open(image_path).convert("RGB")
    tensor = _prediction_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    max_index = int(probabilities.argmax().item())
    predicted_class = class_names[max_index]
    confidence = float(probabilities[max_index].item())

    return {
        "classification": predicted_class,
        "confidence": confidence,
        "class_probabilities": {class_name: float(probabilities[idx].item()) for idx, class_name in enumerate(class_names)},
    }


if __name__ == "__main__":
    result = predict_image("data/raw/classification_demo/DEVELOPING_01_01.png", "models/cyclone_classifier_resnet18.pt")
    print(json.dumps(result, indent=2))
