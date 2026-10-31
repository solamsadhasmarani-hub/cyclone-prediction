

import glob
import os
import pandas as pd
from netCDF4 import Dataset

print("Loading official IBTrACS labels...", flush=True)

csv_path = "data/raw/ibtracs/ibtracs.NI.list.v04r01.csv"
df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)

df["SID"] = df["SID"].astype(str).str.strip()
df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"], errors="coerce"
).dt.round("s")
df["NEWDELHI_GRADE"] = (
    df["NEWDELHI_GRADE"].fillna("").astype(str).str.strip().str.upper()
)

valid = (
    df["ISO_TIME"].notna()
    & df["NEWDELHI_GRADE"].ne("")
    & ~df["NEWDELHI_GRADE"].isin(["NAN", "NONE", "NAT"])
)

lookup = dict(zip(
    zip(df.loc[valid, "SID"], df.loc[valid, "ISO_TIME"]),
    df.loc[valid, "NEWDELHI_GRADE"],
))

print("Labels loaded:", len(lookup), flush=True)

files = glob.glob("data/raw/hursat/**/*.nc", recursive=True)
matched = []
unmatched = []

print("Satellite files found:", len(files), flush=True)

for i, path in enumerate(files, start=1):
    try:
        # Read only metadata; don't load all satellite arrays.
        with Dataset(path, "r") as ds:
            sid = ds.getncattr("TC_serial_number").strip()
            raw_time = ds.variables["htime"][0]

            units = ds.variables["htime"].units
            from netCDF4 import num2date
            timestamp = pd.Timestamp(
                num2date(raw_time, units, only_use_cftime_datetimes=False)
            ).round("s")

        grade = lookup.get((sid, timestamp))
        item = {
            "file": os.path.basename(path),
            "SID": sid,
            "timestamp": timestamp,
            "grade": grade or "",
        }

        if grade:
            matched.append(item)
        else:
            unmatched.append(item)

        if i % 25 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)}", flush=True)

    except Exception as exc:
        print("Could not read:", os.path.basename(path), str(exc), flush=True)

os.makedirs("data/processed/metadata", exist_ok=True)
output = "data/processed/metadata/hursat_ibtracs_match_report.csv"

report = pd.DataFrame(matched)
report.to_csv(output, index=False)

print("\nExact-time matches:", len(matched))
print("Unmatched files:", len(unmatched))
if matched:
    print("\nGrade counts:")
    print(report["grade"].value_counts().to_string())
print("\nReport saved:", output)

