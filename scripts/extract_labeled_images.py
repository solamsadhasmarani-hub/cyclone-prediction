
from pathlib import Path
import pandas as pd

METADATA = Path("data/processed/metadata/hursat_deduplicated.csv")
IMAGE_ROOT = Path("data/processed/images")
OUTPUT = Path("data/processed/metadata/training_manifest.csv")


def main():
    df = pd.read_csv(METADATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    matched_paths = []
    missing_records = []
    duplicate_records = 0

    for _, row in df.iterrows():
        sid = str(row["SID"]).strip()
        grade = str(row["grade"]).strip()
        timestamp = row["timestamp"]

        # Match the filename prefix in the official grade folder.
        prefix = f"{sid}_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_"
        grade_folder = IMAGE_ROOT / grade

        candidates = (
            sorted(grade_folder.glob(f"{prefix}*.png"))
            if grade_folder.is_dir()
            else []
        )

        if candidates:
            # Several files can be duplicates from earlier extraction runs.
            # Select one deterministically for this metadata record.
            matched_paths.append(str(candidates[0].resolve()))

            if len(candidates) > 1:
                duplicate_records += 1
        else:
            matched_paths.append("")
            missing_records.append(
                (sid, str(timestamp), grade)
            )

    df["image_path"] = matched_paths
    valid = df[df["image_path"] != ""].copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(OUTPUT, index=False)

    print("\n--- MANIFEST VERIFICATION ---")
    print("Metadata records:", len(df))
    print("Records matched:", len(valid))
    print("Records missing:", len(missing_records))
    print("Records with duplicate PNGs:", duplicate_records)

    print("\nGrade counts:")
    if not valid.empty:
        print(valid["grade"].value_counts().sort_index().to_string())

    if missing_records:
        print("\nFirst 10 missing records:")
        for record in missing_records[:10]:
            print(record)

    print("\nSaved manifest:", OUTPUT)


if __name__ == "__main__":
    main()
