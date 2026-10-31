from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

REQUIRED_COLUMNS = {
    "image_path",
    "timestamp",
    "latitude",
    "longitude",
    "cyclone_id",
    "cyclone_name",
    "intensity",
    "wind_speed",
    "pressure",
    "classification",
    "is_cyclone",
}


def _resolve_image_path(csv_path: str | Path, image_ref: str | Path) -> Path:
    """Resolve image references that may be relative to the project root or raw-data folder."""
    csv_file = Path(csv_path).resolve()
    image_path = Path(image_ref)
    candidate_roots = [csv_file.parent, *csv_file.parents]

    # Common case: project-relative path, e.g. data/raw/demo_images/example.png
    if not image_path.is_absolute():
        for root in candidate_roots:
            candidate = root / image_path
            if candidate.exists():
                return candidate

            parts = image_path.parts
            if len(parts) >= 2 and parts[0] == "data" and parts[1] == "raw":
                stripped = Path(*parts[2:])
                candidate = root / stripped
                if candidate.exists():
                    return candidate

    return csv_file.parent / image_path if not image_path.is_absolute() else image_path


def detect_corrupted_image(image_path: str | Path) -> bool:
    """Return True if the image is unreadable or missing."""
    path = Path(image_path)
    if not path.exists():
        return True

    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError, ValueError):
        return True

    return False


def validate_metadata_file(csv_path: str | Path) -> pd.DataFrame:
    """Validate metadata presence, schema, and data file availability."""
    csv_path = Path(csv_path).resolve()
    dataframe = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset metadata missing required columns: {sorted(missing)}")

    for _, row in dataframe.iterrows():
        image_path = _resolve_image_path(csv_path, row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Referenced image is missing: {image_path}")
        if detect_corrupted_image(image_path):
            raise ValueError(f"Corrupted image detected: {image_path}")

    return dataframe


def validate_split_no_leakage(dataframe: pd.DataFrame, train_ids: set[str], val_ids: set[str], test_ids: set[str]) -> bool:
    """Ensure no cyclone ID appears in more than one split."""
    all_split_ids = train_ids | val_ids | test_ids
    if len(all_split_ids) != len(dataframe["cyclone_id"].unique()):
        return False

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    return len(overlap) == 0
