import pandas as pd
from pathlib import Path

source = Path("data/raw/detection_demo_metadata.csv")
output = Path("data/processed/metadata/detection_demo_metadata_fixed.csv")
image_dir = Path("data/raw/detection_demo")

df = pd.read_csv(source)
df["image_path"] = df["image_path"].fillna("").astype(str).str.strip()

for i, row in df.iterrows():
    if not row["image_path"] or row["image_path"].lower() == "nan":
        if str(row.get("event_type", "")).strip().lower() == "non_cyclone":
            candidates = sorted(image_dir.glob("NO_CYC_*.png"))
        else:
            candidates = sorted(image_dir.glob("CYC_*.png"))

        if candidates:
            # Assign a unique image of the corresponding type.
            used = set(df.loc[:i-1, "image_path"].dropna().astype(str)) if i > 0 else set()
            available = [p for p in candidates if str(p).replace("\\", "/") not in used]
            if available:
                df.at[i, "image_path"] = str(available[0]).replace("\\", "/")

df["image_path"] = df["image_path"].apply(
    lambda p: str((Path("data") / p).resolve()) if p and not Path(p).is_absolute() and not Path(p).exists() else str(Path(p).resolve()) if p else ""
)
df.to_csv(output, index=False)

print("Saved:", output)
print("Rows:", len(df))
print("Blank paths:", (df["image_path"].str.strip() == "").sum())
print("Missing files:", sum(bool(p) and not Path(p).is_file() for p in df["image_path"]))
print(df.groupby(["label", "event_type"]).size().to_string())
