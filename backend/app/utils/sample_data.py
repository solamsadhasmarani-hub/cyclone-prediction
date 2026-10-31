from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw

from backend.app.config import settings


def create_demo_cyclone_image(output_path: Path, center_x: int, center_y: int, ring_index: int) -> None:
    """Create a synthetic cyclone-like satellite image to support local development."""
    width, height = 224, 224
    image = Image.new("RGB", (width, height), color=(7, 18, 38))
    draw = ImageDraw.Draw(image)

    # Background ocean tones
    for i in range(0, 50, 10):
        draw.ellipse((i, i, width - i, height - i), outline=(22, 58, 84), width=1)

    # Synthetic cloud/rainband pattern
    for ring in range(8):
        radius = 18 + ring * 12 + ring_index * 2
        band = 12 + ring * 4
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=(90 + ring * 10, 140 + ring * 8, 170 + ring * 10),
            width=band,
        )

    # Core storm center
    draw.ellipse(
        (center_x - 18, center_y - 18, center_x + 18, center_y + 18),
        fill=(216, 214, 255),
        outline=(255, 255, 255),
        width=2,
    )

    image.save(output_path)


def generate_demo_dataset(
    project_root: str | Path | None = None,
    cyclone_count: int = 3,
    frames_per_cyclone: int = 5,
) -> tuple[Path, Path]:
    """Create a small synthetic dataset for local development and tests.

    This is intentionally synthetic and clearly not an authoritative or real meteorological dataset.
    """
    root = Path(project_root) if project_root is not None else settings.project_root
    raw_dir = root / "data" / "raw"
    image_dir = raw_dir / "demo_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = raw_dir / "demo_metadata.csv"
    records: list[dict[str, str | int | float]] = []

    for cyclone_index in range(cyclone_count):
        cyclone_id = f"CYCLONE_{cyclone_index + 1:02d}"
        cyclone_name = f"Demo Cyclone {cyclone_index + 1}"
        for frame_index in range(frames_per_cyclone):
            image_name = f"{cyclone_id}_{frame_index + 1:02d}.png"
            image_path = image_dir / image_name
            center_x = 112 + (frame_index - 2) * 8
            center_y = 110 + (cyclone_index * 4)
            create_demo_cyclone_image(image_path, center_x, center_y, frame_index)

            records.append(
                {
                    "image_path": str(Path("data/raw/demo_images") / image_name),
                    "timestamp": f"2024-01-{10 + cyclone_index:02d}T{12 + frame_index:02d}:00:00Z",
                    "latitude": 12.0 + cyclone_index * 2.5 + frame_index * 0.25,
                    "longitude": 80.0 + cyclone_index * 1.8 + frame_index * 0.2,
                    "cyclone_id": cyclone_id,
                    "cyclone_name": cyclone_name,
                    "intensity": 32 + cyclone_index * 12 + frame_index * 4,
                    "wind_speed": 40 + cyclone_index * 14 + frame_index * 5,
                    "pressure": 1005 - cyclone_index * 7 - frame_index * 2,
                    "classification": ["developing", "mature", "weakening"][frame_index % 3],
                    "is_cyclone": 1,
                }
            )

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
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
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return metadata_path, image_dir


def ensure_demo_dataset(project_root: str | Path | None = None) -> tuple[Path, Path]:
    """Create the demo dataset only if it does not already exist."""
    root = Path(project_root) if project_root is not None else settings.project_root
    metadata_path = root / "data" / "raw" / "demo_metadata.csv"
    if not metadata_path.exists():
        return generate_demo_dataset(root)
    return metadata_path, root / "data" / "raw" / "demo_images"


if __name__ == "__main__":
    metadata_path, image_dir = generate_demo_dataset()
    print(f"Created demo metadata: {metadata_path}")
    print(f"Created demo image dir: {image_dir}")
    print(f"Images generated: {len(list(image_dir.glob('*.png')))}")
