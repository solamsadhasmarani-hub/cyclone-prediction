from pathlib import Path
import pandas as pd

metadata = Path("data/processed/metadata/hursat_deduplicated.csv")
image_root = Path("data/processed/images")
output = Path("data/processed/metadata/training_manifest.csv")

df = pd.read_csv(metadata)
df["timestamp"] = pd.to_datetime(df["timestamp"])

png_files = list(image_root.rglob("*.png"))
print("PNG files discovered:", len(png_files))

matched = []
missing = []

for _, row in df.iterrows():
    sid = str(row["SID"]).strip()
    grade = str(row["grade"]).strip()
    timestamp = row["timestamp"]

    prefix = f"{sid}_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_"
    folder = image_root / grade

    candidates = sorted(
        p for p in folder.glob("*.png")
        if p.name.startswith(prefix)
    ) if folder.is_dir() else []

    if candidates:
        matched.append(str(candidates[0].resolve()))
    else:
        matched.append("")
        missing.append((sid, str(timestamp), grade))

df["image_path"] = matched
valid = df[df["image_path"] != ""].copy()

output.parent.mkdir(parents=True, exist_ok=True)
valid.to_csv(output, index=False)

print("Metadata records:", len(df))
print("Matched records:", len(valid))
print("Missing records:", len(missing))
print("\nGrade counts:")
if not valid.empty:
    print(valid["grade"].value_counts().sort_index().to_string())

if missing:
    print("\nFirst 10 missing records:")
    for item in missing[:10]:
        print(item)

print("\nManifest saved to:", output)
