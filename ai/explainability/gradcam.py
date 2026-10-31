from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


class GradCAM:
    """A compact Grad-CAM implementation for a ResNet-style classifier."""

    def __init__(self, model: torch.nn.Module, target_layer: str = "layer4"):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        target_module = self._resolve_module(target_layer)
        self._forward_handle = target_module.register_forward_hook(self._forward_hook)
        self._backward_handle = target_module.register_full_backward_hook(self._backward_hook)

    def _resolve_module(self, name: str):
        if not hasattr(self.model, name):
            raise AttributeError(f"Model does not have layer '{name}'")
        return getattr(self.model, name)

    def _forward_hook(self, module, input_tensor, output):
        self.feature_maps = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        try:
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
            return cam.squeeze(0).squeeze(0).cpu().numpy()
        finally:
            self._forward_handle.remove()
            self._backward_handle.remove()


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
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = _prediction_transform()(image).unsqueeze(0)
    gradcam = GradCAM(model)
    return gradcam.generate(tensor, class_index=class_index)


def explain_prediction(model: torch.nn.Module, image_path: str | Path, class_names: list[str] | None = None) -> dict[str, object]:
    """Produce a simplified Grad-CAM explanation object for a prediction."""
    if class_names is None:
        class_names = ["developing", "mature", "weakening", "dissipating"]

    model.eval()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = _prediction_transform()(image).unsqueeze(0)
    with torch.no_grad():
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
    print("Grad-CAM helper ready.")
