
import glob
import os
import pandas as pd
import xarray as xr

csv_path = "data/raw/ibtracs/ibtracs.NI.list.v04r01.csv"
df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)

df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

print("Official IBTrACS labels for NADA and VARDAH:\n")

storms = ["NADA", "VARDAH"]

for storm in storms:
    rows = df[df["NAME"].astype(str).str.upper() == storm]
    print(f"\n{storm}: {len(rows)} records")
    print(
        rows[["NAME", "ISO_TIME", "NEWDELHI_GRADE", "NEWDELHI_WIND"]]
        .head(8)
        .to_string(index=False)
    )

print("\nHURSAT satellite file timestamps:\n")

files = glob.glob("data/raw/hursat/**/*.nc", recursive=True)

for path in files:
    name = os.path.basename(path)
    if not any(storm in name.upper() for storm in storms):
        continue

    with xr.open_dataset(path) as ds:
        timestamp = pd.to_datetime(ds["htime"].values[0])

    print(name, "->", timestamp)
