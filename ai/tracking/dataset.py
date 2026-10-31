from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def generate_track_demo_dataset(project_root: str | Path | None = None, events: int = 4) -> Path:
    """Generate synthetic cyclone track data for local demo development only."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    start_time = datetime(2024, 1, 1, 0, 0)

    for event_index in range(1, events + 1):
        cyclone_id = f"TRACK_{event_index:02d}"
        latitude = 10.0 + event_index * 1.5
        longitude = 80.0 + event_index * 1.2

        for step in range(1, 15):
            timestamp = start_time + timedelta(hours=step * 6)
            lat_shift = step * 0.25 + event_index * 0.12
            lon_shift = step * 0.35 + event_index * 0.18
            future_latitude = latitude + lat_shift
            future_longitude = longitude + lon_shift
            rows.append(
                {
                    "timestamp": timestamp.isoformat(timespec="hours") + "Z",
                    "cyclone_id": cyclone_id,
                    "latitude": round(latitude + step * 0.4, 3),
                    "longitude": round(longitude + step * 0.5, 3),
                    "hours_since_start": step * 6,
                    "prediction_horizon_hours": 12,
                    "future_latitude": round(future_latitude, 3),
                    "future_longitude": round(future_longitude, 3),
                }
            )

    metadata_path = raw_dir / "track_demo_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp",
                "cyclone_id",
                "latitude",
                "longitude",
                "hours_since_start",
                "prediction_horizon_hours",
                "future_latitude",
                "future_longitude",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return metadata_path


def load_track_dataset(metadata_path: str | Path) -> pd.DataFrame:
    """Load track data and validate required columns."""
    dataframe = pd.read_csv(metadata_path)
    required = {
        "timestamp",
        "cyclone_id",
        "latitude",
        "longitude",
        "hours_since_start",
        "prediction_horizon_hours",
        "future_latitude",
        "future_longitude",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Track dataset missing required columns: {sorted(missing)}")
    return dataframe


if __name__ == "__main__":
    metadata_path = generate_track_demo_dataset()
    print(f"Track demo dataset: {metadata_path}")
    print(f"Rows: {len(load_track_dataset(metadata_path))}")
