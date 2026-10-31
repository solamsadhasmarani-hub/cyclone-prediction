
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/metadata/training_manifest.csv"
OUTPUT_DIR = ROOT / "data/processed/metadata"

df = pd.read_csv(MANIFEST)

required = {"SID", "grade", "image_path"}
missing_columns = required - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")

# Confirm all image files exist.
df["image_path"] = df["image_path"].map(
    lambda p: str(Path(p).resolve())
)
bad_paths = df[~df["image_path"].map(lambda p: Path(p).is_file())]
if not bad_paths.empty:
    raise FileNotFoundError(
        f"{len(bad_paths)} image paths do not exist."
    )

# Hold out one complete cyclone for validation.
# This is a diagnostic split, not a balanced benchmark.
validation_sid = "2016341N08092"  # VARDAH

train = df[df["SID"] != validation_sid].copy()
validation = df[df["SID"] == validation_sid].copy()

if train.empty or validation.empty:
    raise ValueError("Training or validation split is empty.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = OUTPUT_DIR / "train_manifest.csv"
validation_path = OUTPUT_DIR / "validation_manifest.csv"

train.to_csv(train_path, index=False)
validation.to_csv(validation_path, index=False)

print("Split created successfully")
print("Training images:", len(train))
print("Validation images:", len(validation))
print("Training cyclones:", sorted(train["SID"].unique()))
print("Validation cyclone:", validation_sid)

print("\nTraining grade counts:")
print(train["grade"].value_counts().sort_index().to_string())

print("\nValidation grade counts:")
print(validation["grade"].value_counts().sort_index().to_string())

print("\nSaved files:")
print(train_path)
print(validation_path)

print(
    "\nWarning: this split cannot validate grades absent "
    "from the held-out cyclone. It is not a complete "
    "seven- or eight-class evaluation."
)
