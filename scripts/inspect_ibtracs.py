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

    try:
        dataframe = load_ibtracs_csv(csv_path)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(f"Could not read IBTrACS CSV {csv_path.name}: {error}")
        return 2

    print(f"Filename: {csv_path.name}")
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns):,}")
    print("All column names:")
    for column in dataframe.columns:
        print(f"  - {column}")

    possible_fields = identify_columns(dataframe.columns)
    print("\nPossible field columns (candidates, not verified selections):")
    for role, columns in possible_fields.items():
        print(f"  {role}: {', '.join(columns) if columns else 'not found'}")

    important_columns = list(dict.fromkeys(
        column for columns in possible_fields.values() for column in columns
    ))
    if important_columns:
        print("\nMissing-value statistics for candidate important columns:")
        missing = dataframe[important_columns].isna().sum()
        for column in important_columns:
            percent = (float(missing[column]) / len(dataframe) * 100) if len(dataframe) else 0.0
            print(f"  {column}: {missing[column]:,} missing ({percent:.2f}%)")
    else:
        print("\nNo candidate important columns were identified.")

    agency_candidates = possible_fields["agency/source"]
    if agency_candidates:
        print("\nUnique agencies/sources by candidate column:")
        for column in agency_candidates:
            values = dataframe[column].dropna().astype(str).unique().tolist()
            print(f"  {column}: {values}")
    else:
        print("\nUnique agencies/sources: no candidate agency/source column found.")

    print("\nFirst 5 rows:")
    print(dataframe.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
