from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ai.classification.dataset import CLASS_NAMES, ClassificationDataset, generate_classification_demo_dataset
from ai.classification.model import build_classifier_model


def _get_transform(train: bool = False):
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        transform_list = [transforms.RandomHorizontalFlip(p=0.5), *transform_list]
    return transforms.Compose(transform_list)


def train_classifier(project_root: str | Path | None = None, epochs: int = 8, batch_size: int = 8, learning_rate: float = 1e-4) -> dict[str, object]:
    """Train a cyclone stage classifier on the synthetic demo dataset."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    metadata_path = generate_classification_demo_dataset(root)

    train_dataset = ClassificationDataset(metadata_path, split="train", transform=_get_transform(train=True))
    validation_dataset = ClassificationDataset(metadata_path, split="validation", transform=_get_transform(train=False))
    test_dataset = ClassificationDataset(metadata_path, split="test", transform=_get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier_model(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_images.size(0)
            train_total += batch_labels.size(0)
            train_correct += (logits.argmax(dim=1) == batch_labels).sum().item()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in validation_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                logits = model(val_images)
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_accuracy = val_correct / val_total if val_total else 0.0
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss / train_total:.4f} | train_acc={train_correct / train_total:.4f} | val_acc={val_accuracy:.4f}")

    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / "cyclone_classifier_resnet18.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "best_val_accuracy": best_val_accuracy,
    }, model_path)

    eval_summary = evaluate_classifier(model_path, metadata_path)
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "best_validation_accuracy": best_val_accuracy,
        "evaluation": eval_summary,
    }


def evaluate_classifier(model_path: str | Path, metadata_path: str | Path) -> dict[str, object]:
    """Evaluate the saved classifier on the synthetic demo test split."""
    checkpoint = torch.load(model_path, map_location="cpu")
    model = build_classifier_model(num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = ClassificationDataset(metadata_path, split="test", transform=_get_transform(False))
    loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    all_pred = []
    all_true = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            all_pred.extend(predictions.tolist())
            all_true.extend(labels.tolist())

    labels = np.array(all_true)
    preds = np.array(all_pred)
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(labels, preds, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "accuracy": float((preds == labels).mean()),
        "class_names": CLASS_NAMES,
    }


if __name__ == "__main__":
    result = train_classifier()
    print(json.dumps(result, indent=2))
