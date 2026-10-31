from pathlib import Path

import numpy as np

from backend.app.utils.preprocessing import preprocess_image, split_dataset_by_cyclone
from backend.app.utils.sample_data import generate_demo_dataset
from backend.app.utils.validation import validate_metadata_file


def test_phase1_pipeline(tmp_path: Path):
    metadata_path, image_dir = generate_demo_dataset(project_root=tmp_path, cyclone_count=2, frames_per_cyclone=3)
    dataframe = validate_metadata_file(metadata_path)

    assert len(dataframe) == 6
    assert {"image_path", "cyclone_id", "classification"}.issubset(dataframe.columns)

    sample_row = dataframe.iloc[0]
    image_file = tmp_path / sample_row["image_path"].replace("data/raw/demo_images", "data/raw/demo_images")
    processed = preprocess_image(image_file, target_size=(224, 224))
    assert processed.shape == (224, 224, 3)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0

    split = split_dataset_by_cyclone(dataframe)
    train_ids = set(split["train"]["cyclone_id"])
    val_ids = set(split["validation"]["cyclone_id"])
    test_ids = set(split["test"]["cyclone_id"])

    assert (train_ids & val_ids) == set()
    assert (train_ids & test_ids) == set()
    assert (val_ids & test_ids) == set()
    assert image_dir.exists()
    assert len(list(image_dir.glob("*.png"))) == 6
    assert not np.isnan(processed).any()
