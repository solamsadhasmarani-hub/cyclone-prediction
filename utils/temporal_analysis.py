from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from ai.config.imd_classes import IMD_CLASSES


CLASS_RANK = {name: index for index, name in enumerate(IMD_CLASSES)}


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else parsed


def timestamp_from_filename(filename: str) -> str | None:
    """Find an ISO-like date or compact YYYYMMDD[HHMM[SS]] timestamp in a filename."""
    import re

    candidates = (
        re.search(
            r"(?<!\d)(20\d{2})-([01]\d)-([0-3]\d)"
            r"(?:[T _-]([0-2]\d)[:_-]([0-5]\d)(?:[:_-]([0-5]\d))?)?",
            filename,
        ),
        re.search(
            r"(?<!\d)(20\d{2})([01]\d)([0-3]\d)"
            r"(?:T?([0-2]\d)([0-5]\d)([0-5]\d)?)?",
            filename,
        ),
    )
    for match in candidates:
        if not match:
            continue
        groups = match.groups()
        try:
            year, month, day = map(int, groups[:3])
            hour, minute, second = (int(part or 0) for part in groups[3:6])
            return datetime(year, month, day, hour, minute, second).isoformat() + "Z"
        except ValueError:
            continue
    return None


def sort_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
        timestamp = parse_timestamp(item.get("timestamp"))
        return (1, 0, str(item.get("filename", ""))) if timestamp is None else (0, timestamp.value, str(item.get("filename", "")))

    return sorted(observations, key=sort_key)


def derive_evolution_status(observations: list[dict[str, Any]]) -> str:
    ordered = [
        item for item in sort_observations(observations)
        if parse_timestamp(item.get("timestamp")) is not None
        and item.get("predicted_class") in CLASS_RANK
    ]
    if len(ordered) < 2:
        return "INSUFFICIENT TEMPORAL DATA"

    previous = CLASS_RANK[ordered[-2]["predicted_class"]]
    current = CLASS_RANK[ordered[-1]["predicted_class"]]
    if current > previous:
        return "STRENGTHENING"
    if current < previous:
        return "WEAKENING"
    return "PEAK"
