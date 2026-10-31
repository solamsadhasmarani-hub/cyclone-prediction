from __future__ import annotations

from pathlib import Path

from backend.app.config import settings
from backend.app.utils.preprocessing import preprocess_image, split_dataset_by_cyclone
from backend.app.utils.sample_data import ensure_demo_dataset
from backend.app.utils.validation import validate_metadata_file


def main() -> None:
    project_root = settings.project_root
    metadata_path, _ = ensure_demo_dataset(project_root)
    dataframe = validate_metadata_file(metadata_path)
    split = split_dataset_by_cyclone(dataframe)

    sample_row = dataframe.iloc[0]
    processed_array = preprocess_image(Path(sample_row["image_path"]))

    print(f"Project root: {project_root}")
    print(f"Metadata rows: {len(dataframe)}")
    print(f"Cyclone IDs: {sorted(dataframe['cyclone_id'].unique().tolist())}")
    print(f"Train rows: {len(split['train'])}")
    print(f"Validation rows: {len(split['validation'])}")
    print(f"Test rows: {len(split['test'])}")
    print(f"Processed image shape: {processed_array.shape}")
    print("Phase 1 pipeline: OK")


if __name__ == "__main__":
    main()
