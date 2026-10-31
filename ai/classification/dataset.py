from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


CLASS_NAMES = ["developing", "mature", "weakening", "dissipating"]


def _create_class_demo_image(output_path: Path, class_name: str, frame_index: int, center_x: int, center_y: int) -> None:
    """Create a synthetic image that visually corresponds to a nominal cyclone stage."""
    width, height = 224, 224
    image = Image.new("RGB", (width, height), color=(4, 12, 28))
    draw = ImageDraw.Draw(image)

    for i in range(0, 70, 10):
        draw.ellipse((i, i, width - i, height - i), outline=(25, 45, 70), width=1)

    stage_profiles = {
        "developing": {"rings": 5, "radius": 18, "color": (95, 165, 210), "center": (205, 230, 255)},
        "mature": {"rings": 8, "radius": 22, "color": (115, 200, 145), "center": (230, 240, 255)},
        "weakening": {"rings": 6, "radius": 17, "color": (175, 170, 120), "center": (220, 220, 200)},
        "dissipating": {"rings": 4, "radius": 14, "color": (170, 140, 125), "center": (210, 220, 198)},
    }

    profile = stage_profiles[class_name]

    for ring in range(profile["rings"]):
        radius = profile["radius"] + ring * 10 + (frame_index % 2) * 4
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=profile["color"],
            width=4 + ring,
        )

    draw.ellipse((center_x - 16, center_y - 16, center_x + 16, center_y + 16), fill=profile["center"], outline=(255, 255, 255), width=2)
    image.save(output_path)


def _assign_split(event_ids: list[str]) -> dict[str, str]:
    event_to_split: dict[str, str] = {}
    shuffled = event_ids[:]
    random.Random(42).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)
    for idx, event_id in enumerate(shuffled):
        if idx < train_end:
            event_to_split[event_id] = "train"
        elif idx < val_end:
            event_to_split[event_id] = "validation"
        else:
            event_to_split[event_id] = "test"
    return event_to_split


def generate_classification_demo_dataset(
    project_root: str | Path | None = None,
    events_per_class: int = 6,
    frames_per_event: int = 3,
) -> Path:
    """Create a synthetic multi-class cyclone classification dataset with grouped splits."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    image_dir = raw_dir / "classification_demo"
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    all_event_ids: list[str] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        for event_index in range(1, events_per_class + 1):
            event_id = f"{class_name.upper()}_{event_index:02d}"
            all_event_ids.append(event_id)
            split = "train"
            for frame_index in range(frames_per_event):
                image_name = f"{event_id}_{frame_index + 1:02d}.png"
                image_path = image_dir / image_name
                center_x = 104 + (frame_index * 8)
                center_y = 108 + (class_index * 8)
                _create_class_demo_image(image_path, class_name, frame_index, center_x, center_y)
                records.append(
                    {
                        "image_path": str(image_path.relative_to(root)),
                        "class_name": class_name,
                        "label": class_index,
                        "event_id": event_id,
                        "split": split,
                    }
                )

    # Reassign split after creation to preserve event-level grouping.
    split_map = _assign_split(all_event_ids)
    for record in records:
        record["split"] = split_map[str(record["event_id"])]

    metadata_path = raw_dir / "classification_demo_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "class_name", "label", "event_id", "split"])
        writer.writeheader()
        writer.writerows(records)

    return metadata_path


class ClassificationDataset:
    """Image dataset for stage classification training and evaluation."""

    def __init__(self, metadata_path: str | Path, split: str, transform=None):
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.transform = transform
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, object]]:
        dataframe = pd.read_csv(self.metadata_path)
        filtered = dataframe[dataframe["split"] == self.split]
        return filtered.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        image_ref = str(row["image_path"])
        metadata_root = self.metadata_path.resolve().parents[2]
        candidate_paths = [
            metadata_root / image_ref,
            metadata_root / "data" / image_ref,
            metadata_root / "data" / "raw" / image_ref,
            Path(image_ref),
        ]
        image_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


if __name__ == "__main__":
    metadata_path = generate_classification_demo_dataset()
    print(f"Classification dataset metadata: {metadata_path}")
    print(f"Records: {len(pd.read_csv(metadata_path))}")
