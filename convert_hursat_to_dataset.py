import xarray as xr
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent

IBTRACS = ROOT / "data" / "raw" / "ibtracs" / "ibtracs.NI.list.v04r01.csv"
IMAGE_DIR = ROOT / "data" / "processed" / "images"
LABEL_FILE = ROOT / "data" / "metadata" / "labels.csv"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)

label_map = {
    "LPA": "Low Pressure Area",
    "D": "Depression",
    "DD": "Deep Depression",
    "CS": "Cyclonic Storm",
    "SCS": "Severe Cyclonic Storm",
    "VSCS": "Very Severe Cyclonic Storm",
    "ESCS": "Extremely Severe Cyclonic Storm",
    "SUCS": "Super Cyclonic Storm",
}

print("Reading IBTrACS...")
df = pd.read_csv(IBTRACS, skiprows=[1], low_memory=False)

df["SID"] = df["SID"].astype(str)
df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
df["imd_class"] = (
    df["NEWDELHI_GRADE"]
    .astype(str)
    .str.strip()
    .map(label_map)
)

storm_dirs = [
    ROOT / "data" / "raw" / "hursat" / "HURSAT_b1_v06_2016335N07088_NADA_c20180912",
    ROOT / "data" / "raw" / "hursat" / "HURSAT_b1_v06_2016341N08092_VARDAH_c20180912",
]

records = []

for storm_dir in storm_dirs:

    nc_files = sorted(storm_dir.glob("*HIM-8*.nc"))

    print()
    print("========================================")
    print("Storm:", storm_dir.name)
    print("HIM-8 files:", len(nc_files))
    print("========================================")

    for nc_file in nc_files:

        parts = nc_file.name.split(".")

        # Example:
        # 2016341N08092.VARDAH.2016.12.06.0900.57.HIM-8...

        sid = parts[0]
        year = int(parts[2])
        month = int(parts[3])
        day = int(parts[4])
        hhmm = parts[5]

        timestamp = pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=int(hhmm[:2]),
            minute=int(hhmm[2:])
        )

        match = df[
            (df["SID"] == sid) &
            (df["ISO_TIME"] == timestamp)
        ]

        if match.empty:
            continue

        imd_class = match.iloc[0]["imd_class"]

        if pd.isna(imd_class):
            continue

        print(timestamp, "->", imd_class)

        with xr.open_dataset(nc_file) as ds:
            arr = ds["IRWIN"].isel(htime=0).values.astype(np.float32)

        arr = np.nan_to_num(
            arr,
            nan=float(np.nanmedian(arr))
        )

        # Consistent brightness-temperature scaling.
        arr = np.clip(arr, 180.0, 310.0)
        arr = ((arr - 180.0) / 130.0 * 255.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        image_name = (
            f"{sid}_"
            f"{timestamp.strftime('%Y%m%d_%H%M')}.png"
        )

        image_path = IMAGE_DIR / image_name

        Image.fromarray(arr, mode="L").save(image_path)

        records.append({
            "image_path": f"data/processed/images/{image_name}",
            "group_id": sid,
            "cyclone_detected": 1,
            "imd_class": imd_class
        })

result = pd.DataFrame(records)

result = result.drop_duplicates(
    subset=["image_path"]
).sort_values(
    ["group_id", "image_path"]
)

result.to_csv(LABEL_FILE, index=False)

print()
print("========================================")
print("DATASET COMPLETE")
print("========================================")
print("Total images:", len(result))
print()
print("Storms:")
print(result["group_id"].value_counts())
print()
print("IMD classes:")
print(result["imd_class"].value_counts())
print()
print("Labels:", LABEL_FILE)
