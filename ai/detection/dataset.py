from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def create_positive_demo_image(output_path: Path, center_x: int, center_y: int, ring_index: int) -> None:
    """Create a synthetic cyclone-like image for positive labels."""
    width, height = 224, 224
    image = Image.new("RGB", (width, height), color=(7, 18, 38))
    draw = ImageDraw.Draw(image)

    for i in range(0, 60, 10):
        draw.ellipse((i, i, width - i, height - i), outline=(26, 58, 84), width=1)

    for ring in range(8):
        radius = 18 + ring * 12 + ring_index * 2
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=(90 + ring * 10, 150 + ring * 8, 180 + ring * 10),
            width=10 + ring * 3,
        )

    draw.ellipse((center_x - 18, center_y - 18, center_x + 18, center_y + 18), fill=(216, 214, 255), outline=(255, 255, 255), width=2)
    image.save(output_path)


def create_negative_demo_image(output_path: Path, seed: int) -> None:
    """Create a synthetic non-cyclone image for negative labels."""
    width, height = 224, 224
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), color=(5, 14, 25))
    draw = ImageDraw.Draw(image)

    for _ in range(25):
        x1 = rng.randint(0, width)
        y1 = rng.randint(0, height)
        x2 = x1 + rng.randint(15, 45)
        y2 = y1 + rng.randint(15, 45)
        color = (rng.randint(40, 120), rng.randint(70, 150), rng.randint(120, 220))
        draw.ellipse((x1, y1, x2, y2), outline=color, width=2)

    image.save(output_path)


def _assign_split(event_ids: list[str], train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict[str, str]:
    event_to_split: dict[str, str] = {}
    shuffled = event_ids[:]
    random.Random(42).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    for idx, event_id in enumerate(shuffled):
        if idx < train_end:
            event_to_split[event_id] = "train"
        elif idx < val_end:
            event_to_split[event_id] = "validation"
        else:
            event_to_split[event_id] = "test"
    return event_to_split


def _assign_balanced_split(event_ids: list[str], label: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict[str, str]:
    shuffled = event_ids[:]
    random.Random(42 + label).shuffle(shuffled)
    total = len(shuffled)
    if total == 1:
        return {shuffled[0]: "train"}
    train_end = max(1, int(total * train_ratio))
    val_end = train_end + max(1, int(total * val_ratio))
    assignments: dict[str, str] = {}
    for idx, event_id in enumerate(shuffled):
        if idx < train_end:
            assignments[event_id] = "train"
        elif idx < val_end:
            assignments[event_id] = "validation"
        else:
            assignments[event_id] = "test"
    return assignments


def generate_detection_demo_dataset(
    project_root: str | Path | None = None,
    positive_events: int = 3,
    negative_events: int = 2,
    frames_per_event: int = 3,
) -> Path:
    """Create a synthetic cyclone/non-cyclone detection dataset with event-based splits."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    image_dir = raw_dir / "detection_demo"
    image_dir.mkdir(parents=True, exist_ok=True)

    positive_event_ids = [f"CYC_{index:02d}" for index in range(1, positive_events + 1)]
    negative_event_ids = [f"NO_CYC_{index:02d}" for index in range(1, negative_events + 1)]

    split_map = {
        **_assign_balanced_split(positive_event_ids, label=1),
        **_assign_balanced_split(negative_event_ids, label=0),
    }
    records: list[dict[str, object]] = []

    for event_id in positive_event_ids:
        split = split_map[event_id]
        for frame_index in range(frames_per_event):
            image_name = f"{event_id}_{frame_index + 1:02d}.png"
            image_path = image_dir / image_name
            create_positive_demo_image(image_path, x_center := 112 + frame_index * 8, 110 + (frame_index % 2) * 10, frame_index)
            records.append(
                {
                    "image_path": str(image_path.relative_to(root)),
                    "label": 1,
                    "event_id": event_id,
                    "split": split,
                    "event_type": "cyclone",
                }
            )

    for event_id in negative_event_ids:
        split = split_map[event_id]
        for frame_index in range(frames_per_event):
            image_name = f"{event_id}_{frame_index + 1:02d}.png"
            image_path = image_dir / image_name
            create_negative_demo_image(image_path, seed=(frame_index + 1) * 31 + len(event_id))
            records.append(
                {
                    "image_path": str(image_path.relative_to(root)),
                    "label": 0,
                    "event_id": event_id,
                    "split": split,
                    "event_type": "non_cyclone",
                }
            )

    metadata_path = raw_dir / "detection_demo_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "label", "event_id", "split", "event_type"])
        writer.writeheader()
        writer.writerows(records)

    return metadata_path


class DetectionDataset:
    """Simple image dataset wrapper for cyclone detection training and validation."""

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
    metadata_path = generate_detection_demo_dataset()
    print(f"Detection dataset metadata: {metadata_path}")
    print(f"Generated rows: {len(pd.read_csv(metadata_path))}")
