from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def generate_intensity_demo_dataset(project_root: str | Path | None = None, events: int = 4) -> Path:
    """Generate a synthetic demo dataset for cyclone intensity forecasting.

    This data is intentionally synthetic and not based on any real-world event records.
    """
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    base_time = datetime(2024, 1, 1, 0, 0)

    for event_index in range(1, events + 1):
        cyclone_id = f"INTENSITY_{event_index:02d}"
        cyclone_name = f"Demo Cyclone {event_index}"
        base_intensity = 28 + event_index * 12
        base_latitude = 10 + event_index * 2
        base_longitude = 80 + event_index * 1.5

        for step in range(1, 13):
            timestamp = base_time + timedelta(hours=step * 6)
            current_intensity = base_intensity + step * 2 + (event_index - 1) * 3
            wind_speed = 28 + step * 3 + event_index * 5
            pressure = 1012 - current_intensity * 0.8 - step * 0.5
            historical_intensity = max(15, current_intensity - 8 + (event_index % 3))
            future_intensity = current_intensity + 8 + event_index + (step % 3)
            latitude = base_latitude + (step * 0.3) + (event_index * 0.2)
            longitude = base_longitude + (step * 0.5) + (event_index * 0.1)

            rows.append(
                {
                    "timestamp": timestamp.isoformat(timespec="hours") + "Z",
                    "latitude": round(latitude, 3),
                    "longitude": round(longitude, 3),
                    "cyclone_id": cyclone_id,
                    "cyclone_name": cyclone_name,
                    "current_intensity": current_intensity,
                    "wind_speed": wind_speed,
                    "pressure": round(pressure, 2),
                    "historical_intensity": historical_intensity,
                    "time_index_hours": step * 6,
                    "prediction_horizon_hours": 12,
                    "future_intensity": future_intensity,
                }
            )

    metadata_path = raw_dir / "intensity_demo_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp",
                "latitude",
                "longitude",
                "cyclone_id",
                "cyclone_name",
                "current_intensity",
                "wind_speed",
                "pressure",
                "historical_intensity",
                "time_index_hours",
                "prediction_horizon_hours",
                "future_intensity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return metadata_path


def load_intensity_dataset(metadata_path: str | Path) -> pd.DataFrame:
    """Load the demo intensity dataset as a DataFrame."""
    dataframe = pd.read_csv(metadata_path)
    required = {
        "timestamp",
        "latitude",
        "longitude",
        "cyclone_id",
        "current_intensity",
        "wind_speed",
        "pressure",
        "historical_intensity",
        "time_index_hours",
        "prediction_horizon_hours",
        "future_intensity",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Intensity dataset missing required columns: {sorted(missing)}")
    return dataframe


if __name__ == "__main__":
    metadata_path = generate_intensity_demo_dataset()
    print(f"Intensity demo dataset: {metadata_path}")
    print(f"Rows: {len(load_intensity_dataset(metadata_path))}")
