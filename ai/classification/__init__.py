"""Cyclone classification package for Phase 3."""

from .dataset import ClassificationDataset, generate_classification_demo_dataset
from .model import build_classifier_model, load_classifier_model, predict_image

__all__ = [
    "ClassificationDataset",
    "generate_classification_demo_dataset",
    "build_classifier_model",
    "load_classifier_model",
    "predict_image",
    "train_classifier",
]


def __getattr__(name: str):
    if name == "train_classifier":
        from .train import train_classifier

        return train_classifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
