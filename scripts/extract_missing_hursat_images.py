from pathlib import Path
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from PIL import Image

ROOT = Path(".")
META = ROOT / "data/processed/metadata/hursat_deduplicated.csv"
HURSAT = ROOT / "data/raw/hursat"
OUT = ROOT / "data/processed/images"

def main():
    df = pd.read_csv(META)
    made = 0
    skipped = 0
    missing = 0

    for _, row in df.iterrows():
        sid = str(row["SID"]).strip()
        grade = str(row["grade"]).strip()
        ts = pd.to_datetime(row["timestamp"])
        stamp = ts.strftime("%Y-%m-%d_%H-%M-%S")
        folder = OUT / grade
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{sid}_{stamp}_"

        if list(folder.glob(prefix + "*.png")):
            skipped += 1
            continue

        candidates = []
        for p in HURSAT.rglob("*.nc"):
            name = p.name
            if sid in name and ts.strftime("%Y.%m.%d.%H%M") in name:
                candidates.append(p)

        if not candidates:
            missing += 1
            continue

        candidates.sort(key=lambda p: 0 if ".HIM-8." in p.name else 1 if ".MET-7." in p.name else 2)

        try:
            with Dataset(candidates[0]) as ds:
                if "IRWIN" not in ds.variables:
                    missing += 1
                    continue
                arr = np.asarray(ds.variables["IRWIN"][0], dtype=np.float32)
                arr = np.squeeze(arr)
                arr[~np.isfinite(arr)] = np.nan
                lo, hi = np.nanpercentile(arr, [1, 99])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    missing += 1
                    continue
                scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
                pixels = (scaled * 255).astype(np.uint8)
                dest = folder / f"{prefix}0000.png"
                Image.fromarray(pixels).resize((224, 224)).save(dest)
                made += 1
        except Exception as exc:
            print("Failed:", candidates[0].name, str(exc))
            missing += 1

    print("New images created:", made)
    print("Already existed:", skipped)
    print("Records without usable source:", missing)

if __name__ == "__main__":
    main()
