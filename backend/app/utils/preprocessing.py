from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from backend.app.utils.data_loader import load_image
from backend.app.utils.validation import detect_corrupted_image


def resize_image(image_array: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize an image to the target dimensions."""
    resized = Image.fromarray(image_array.astype("uint8")).resize(target_size)
    return np.asarray(resized)


def normalize_image(image_array: np.ndarray) -> np.ndarray:
    """Scale pixels to the [0, 1] range for model inputs."""
    return image_array.astype(np.float32) / 255.0


def preprocess_image(image_path: str | Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Validate, resize, and normalize one image."""
    path = Path(image_path)
    if detect_corrupted_image(path):
        raise ValueError(f"Input image is invalid or corrupted: {path}")

    image_array = load_image(path)
    resized = resize_image(image_array, target_size)
    return normalize_image(resized)


def split_dataset_by_cyclone(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Split the dataset by cyclone ID to avoid leakage across events."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    unique_ids = list(dataframe["cyclone_id"].drop_duplicates())
    random.Random(42).shuffle(unique_ids)

    total_ids = len(unique_ids)
    train_count = int(total_ids * train_ratio)
    val_count = int(total_ids * val_ratio)
    test_count = total_ids - train_count - val_count

    train_ids = set(unique_ids[:train_count])
    val_ids = set(unique_ids[train_count : train_count + val_count])
    test_ids = set(unique_ids[train_count + val_count : train_count + val_count + test_count])

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Split overlap detected; revise group assignments.")

    split_map = {
        "train": dataframe[dataframe["cyclone_id"].isin(train_ids)].copy(),
        "validation": dataframe[dataframe["cyclone_id"].isin(val_ids)].copy(),
        "test": dataframe[dataframe["cyclone_id"].isin(test_ids)].copy(),
    }
    return split_map
