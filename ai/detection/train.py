from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ai.detection.dataset import DetectionDataset, generate_detection_demo_dataset
from ai.detection.model import build_detector_model


def _get_transform(train: bool = False):
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        transform_list = [transforms.RandomHorizontalFlip(p=0.5), *transform_list]
    return transforms.Compose(transform_list)


def train_detector(project_root: str | Path | None = None, epochs: int = 5, batch_size: int = 8, learning_rate: float = 1e-4) -> dict[str, float | str]:
    """Train a simple cyclone detector on the synthetic sample dataset."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    metadata_path = generate_detection_demo_dataset(root)

    train_dataset = DetectionDataset(metadata_path, split="train", transform=_get_transform(train=True))
    validation_dataset = DetectionDataset(metadata_path, split="validation", transform=_get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detector_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_images.size(0)
            total += batch_labels.size(0)
            correct += (logits.argmax(dim=1) == batch_labels).sum().item()

        train_accuracy = correct / total if total else 0.0
        val_accuracy = 0.0
        num_val = 0
        model.eval()
        with torch.no_grad():
            for val_images, val_labels in validation_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                logits = model(val_images)
                predictions = logits.argmax(dim=1)
                correct_val = (predictions == val_labels).sum().item()
                val_accuracy += correct_val
                num_val += val_labels.size(0)
        val_accuracy = val_accuracy / num_val if num_val else 0.0
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} | train_loss={running_loss / total if total else 0.0:.4f} | train_acc={train_accuracy:.4f} | val_acc={val_accuracy:.4f}")

    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / "cyclone_detector_resnet18.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": 2,
        "best_val_accuracy": best_val_accuracy,
        "metadata_path": str(metadata_path),
    }
    torch.save(checkpoint, model_path)

    return {
        "model_path": str(model_path),
        "best_validation_accuracy": best_val_accuracy,
        "metadata_path": str(metadata_path),
    }


if __name__ == "__main__":
    results = train_detector()
    print(json.dumps(results, indent=2))
