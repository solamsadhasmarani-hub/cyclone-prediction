"""Shared, non-destructive helpers for inspecting an IBTrACS CSV."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IBTRACS_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "ibtracs"

_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "storm ID": ("sid", "stormid", "cycloneid", "stormidentifier", "atcfid"),
    "storm name": ("name", "stormname", "cyclonename"),
    "timestamp": ("isotime", "timestamp", "datetime", "time", "obstime", "validtime"),
    "latitude": ("lat", "latitude"),
    "longitude": ("lon", "long", "longitude"),
    "wind speed": ("wind", "windspeed", "maxwind", "maximumwind"),
    "pressure": ("pres", "pressure", "centralpressure", "minpressure"),
    "agency/source": ("agency", "source", "tracktype", "datasource"),
}


def find_ibtracs_csv(directory: Path = IBTRACS_DIRECTORY) -> Path | None:
    """Return the sole CSV in the input directory tree, if present."""
    if not directory.is_dir():
        return None
    csv_files = sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.casefold() == ".csv"),
        key=lambda path: str(path).casefold(),
    )
    if len(csv_files) > 1:
        choices = "\n".join(f"  - {path}" for path in csv_files)
        raise ValueError(f"Expected one IBTrACS CSV; found {len(csv_files)}:\n{choices}")
    return csv_files[0] if csv_files else None


def load_ibtracs_csv(path: Path) -> pd.DataFrame:
    """Read a CSV without modifying it; parsing/encoding errors are surfaced."""
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _normalize_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def identify_columns(columns: object) -> dict[str, list[str]]:
    """Return possible field names, without assuming one field is authoritative."""
    column_names = [str(column) for column in columns]  # type: ignore[union-attr]
    normalized = {column: _normalize_column_name(column) for column in column_names}
    result: dict[str, list[str]] = {}

    for role, aliases in _ROLE_ALIASES.items():
        exact = [
            column for column in column_names
            if normalized[column] in aliases
        ]
        broad = [
            column for column in column_names
            if column not in exact and any(alias in normalized[column] for alias in aliases)
        ]
        result[role] = exact + broad
    return result


def unambiguous_column(columns: object, role: str) -> str | None:
    """Return a sole heuristic candidate, without treating it as authoritative."""
    candidates = identify_columns(columns)[role]
    return candidates[0] if len(candidates) == 1 else None


def missing_column_message(columns: object, role: str) -> str:
    candidates = identify_columns(columns)[role]
    if candidates:
        return f"{role}: ambiguous candidates {candidates}; select the verified field before analysis"
    return f"{role}: no candidate column found"
