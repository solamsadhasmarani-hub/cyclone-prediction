from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.data.ibtracs import (  # noqa: E402
    IBTRACS_DIRECTORY,
    find_ibtracs_csv,
    identify_columns,
    load_ibtracs_csv,
    unambiguous_column,
)


def _report_numeric_check(dataframe: pd.DataFrame, column: str | None, label: str) -> None:
    if column is None:
        print(f"{label}: NOT CHECKED (no unique, exact candidate column)")
        return

    values = dataframe[column]
    numeric = pd.to_numeric(values, errors="coerce")
    non_missing = values.notna()
    invalid = non_missing & numeric.isna()
    role = label.split()[0]
    if role == "latitude":
        invalid |= numeric.notna() & ~numeric.between(-90, 90)
    elif role == "longitude":
        invalid |= numeric.notna() & ~numeric.between(-180, 180)
    elif role == "wind":
        invalid |= numeric.notna() & (numeric < 0)
    elif role == "pressure":
        invalid |= numeric.notna() & (numeric <= 0)

    print(
        f"{label} ({column}): {int(numeric.notna().sum()):,} numeric values; "
        f"{int(values.isna().sum()):,} missing; {int(invalid.sum()):,} invalid"
    )


def main() -> int:
    print(f"IBTrACS directory: {IBTRACS_DIRECTORY}")
    try:
        csv_path = find_ibtracs_csv()
    except ValueError as error:
        print(error)
        return 2

    if csv_path is None:
        print("IBTrACS North Indian dataset is not present. Please place the CSV in data/raw/ibtracs/.")
        return 2

    print(f"File exists: {csv_path}")
    try:
        dataframe = load_ibtracs_csv(csv_path)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(f"CSV readable: NO ({error})")
        return 2

    print(f"CSV readable: YES ({len(dataframe):,} rows, {len(dataframe.columns):,} columns)")
    candidates = identify_columns(dataframe.columns)
    selected = {
        role: unambiguous_column(dataframe.columns, role)
        for role in candidates
    }

    errors = 0
    print("\nRequired information candidate availability (verify against source documentation):")
    for role in ("storm ID", "timestamp", "latitude", "longitude"):
        role_candidates = candidates[role]
        if not role_candidates:
            print(f"  {role}: no candidate field found")
            errors += 1
        else:
            print(f"  {role}: candidate field(s) {role_candidates}")

    for role in ("wind speed", "pressure", "agency/source"):
        column = selected[role]
        if column is None:
            print(f"  {role}: unavailable as a unique, exact field; candidates={candidates[role]}")
        else:
            print(f"  {role}: available as {column}")

    print("\nValue checks (records are reported, never removed):")
    timestamp_column = selected["timestamp"]
    if timestamp_column is None:
        print("timestamps: NOT CHECKED (no unique candidate; review listed columns)")
    else:
        values = dataframe[timestamp_column]
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        invalid_timestamp = values.notna() & parsed.isna()
        print(
            f"timestamps candidate ({timestamp_column}): {int(parsed.notna().sum()):,} parsed; "
            f"{int(values.isna().sum()):,} missing; "
            f"{int(invalid_timestamp.sum()):,} invalid"
        )

    for role, label in (
        ("latitude", "latitude"),
        ("longitude", "longitude"),
        ("wind speed", "wind"),
        ("pressure", "pressure"),
    ):
        role_candidates = candidates[role]
        if not role_candidates:
            _report_numeric_check(dataframe, None, label)
        else:
            for column in role_candidates:
                _report_numeric_check(dataframe, column, f"{label} candidate")

    storm_id_column = selected["storm ID"]
    if storm_id_column is not None and timestamp_column is not None:
        pair = dataframe[[storm_id_column, timestamp_column]]
        complete_pair = pair.notna().all(axis=1)
        duplicate_count = int(pair.loc[complete_pair].duplicated(keep="first").sum())
        print(
            f"Duplicate storm/timestamp records "
            f"({storm_id_column}, {timestamp_column}): {duplicate_count:,}"
        )
    else:
        print("Duplicate storm/timestamp records: NOT CHECKED (storm ID/timestamp unavailable or ambiguous)")

    print(
        "\nWind units, averaging period, and authoritative intensity-field selection "
        "are intentionally not inferred by this validator."
    )
    print(f"Missing/ambiguous required fields: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
