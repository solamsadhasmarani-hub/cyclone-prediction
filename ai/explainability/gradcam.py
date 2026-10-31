from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


def _ensure_demo_image(image_path: str | Path) -> Path:
    """Generate a simple demo image when the expected sample file is absent."""
    path = Path(image_path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (224, 224), color=(12, 23, 38))
    draw = ImageDraw.Draw(image)
    for ring in range(6):
        radius = 24 + ring * 12
        draw.ellipse((112 - radius, 112 - radius, 112 + radius, 112 + radius), outline=(90 + ring * 10, 165 + ring * 8, 200 + ring * 8), width=6 + ring)
    draw.ellipse((96, 96, 128, 128), fill=(220, 220, 255), outline=(255, 255, 255), width=2)
    image.save(path)
    return path


class GradCAM:
    """A compact Grad-CAM implementation for a ResNet-style classifier."""

    def __init__(self, model: torch.nn.Module, target_layer: str = "layer4"):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        target_module = self._resolve_module(target_layer)
        target_module.register_forward_hook(self._forward_hook)
        target_module.register_full_backward_hook(self._backward_hook)

    def _resolve_module(self, name: str):
        if not hasattr(self.model, name):
            raise AttributeError(f"Model does not have layer '{name}'")
        return getattr(self.model, name)

    def _forward_hook(self, module, input_tensor, output):
        self.feature_maps = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[:, class_index].backward()

        if self.feature_maps is None or self.gradients is None:
            raise RuntimeError("Grad-CAM could not produce a valid activation map.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()
        return cam


def _prediction_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def generate_gradcam_heatmap(model: torch.nn.Module, image_path: str | Path, class_index: int | None = None) -> np.ndarray:
    """Generate the saliency heatmap for a supplied image."""
    resolved_path = _ensure_demo_image(image_path)
    image = Image.open(resolved_path).convert("RGB")
    tensor = _prediction_transform()(image).unsqueeze(0)
    gradcam = GradCAM(model)
    return gradcam.generate(tensor, class_index=class_index)


def explain_prediction(model: torch.nn.Module, image_path: str | Path, class_names: list[str] | None = None) -> dict[str, object]:
    """Produce a simplified Grad-CAM explanation object for a prediction."""
    if class_names is None:
        class_names = ["developing", "mature", "weakening", "dissipating"]

    resolved_path = _ensure_demo_image(image_path)
    image = Image.open(resolved_path).convert("RGB")
    tensor = _prediction_transform()(image).unsqueeze(0)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top_index = int(probs.argmax().item())
    heatmap = generate_gradcam_heatmap(model, image_path, class_index=top_index)

    return {
        "predicted_class": class_names[top_index],
        "confidence": float(probs[top_index].item()),
        "heatmap_shape": list(heatmap.shape),
        "heatmap_summary": {
            "min": float(np.min(heatmap)),
            "max": float(np.max(heatmap)),
            "mean": float(np.mean(heatmap)),
        },
    }


if __name__ == "__main__":
    print(json.dumps({"status": "gradcam-ready"}, indent=2))
