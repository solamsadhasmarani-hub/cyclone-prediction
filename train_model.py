from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.config.imd_classes import IMD_CLASSES  # noqa: E402
from utils.preprocessing import PREPROCESSING_CONFIG  # noqa: E402

MANIFEST_PATH = ROOT / "data" / "metadata" / "labels.csv"
MODEL_PATH = ROOT / "models" / "cyclone_mobilenetv2.keras"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
CONFUSION_MATRIX_PATH = ROOT / "outputs" / "charts" / "confusion_matrix.png"
REQUIRED_COLUMNS = {"image_path", "group_id", "cyclone_detected", "imd_class"}


def _parse_detection_label(value: Any, row_number: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return int(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "cyclone"}:
        return 1
    if normalized in {"0", "false", "no", "non-cyclone", "noncyclone"}:
        return 0
    raise ValueError(
        f"Row {row_number}: cyclone_detected must be an explicit 0/1 or true/false value."
    )


def _resolve_image_path(image_ref: str) -> Path:
    path = Path(image_ref)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_manifest(manifest_path: str | Path = MANIFEST_PATH) -> pd.DataFrame:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Real labeled training manifest not found: {path}\n"
            "Add verified rows to data/metadata/labels.csv; do not use demo data."
        )
    try:
        frame = pd.read_csv(path, keep_default_na=True)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read training manifest {path}: {error}") from error

    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Training manifest is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Training manifest is empty; no model will be trained.")

    frame = frame.copy()
    frame["image_path"] = frame["image_path"].astype("string").str.strip()
    frame["group_id"] = frame["group_id"].astype("string").str.strip()
    if frame["image_path"].isna().any() or (frame["image_path"] == "").any():
        raise ValueError("Every row must include an image_path.")
    if frame["group_id"].isna().any() or (frame["group_id"] == "").any():
        raise ValueError("Every row must include a group_id for leakage-safe splitting.")
    resolved_paths = frame["image_path"].map(lambda value: str(_resolve_image_path(str(value))))
    if resolved_paths.duplicated().any():
        duplicate = resolved_paths[resolved_paths.duplicated(keep=False)].iloc[0]
        raise ValueError(f"Duplicate image paths are not allowed in the training manifest: {duplicate}")

    frame["cyclone_detected"] = [
        _parse_detection_label(value, row + 2)
        for row, value in enumerate(frame["cyclone_detected"])
    ]
    frame["imd_class"] = frame["imd_class"].astype("string").str.strip()
    for row_index, row in frame.iterrows():
        image_path = _resolve_image_path(str(row["image_path"]))
        if any(part.casefold() in {"classification_demo", "detection_demo", "demo_images"} for part in image_path.parts):
            raise ValueError(
                f"Demo data is not permitted for training (manifest row {row_index + 2}: {image_path})."
            )
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist (manifest row {row_index + 2}): {image_path}")
        if image_path.suffix.casefold() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Unsupported image type at manifest row {row_index + 2}: {image_path.name}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(f"Invalid image at manifest row {row_index + 2}: {image_path}") from error
        frame.at[row_index, "image_path"] = str(image_path)

        class_name = row["imd_class"]
        if int(row["cyclone_detected"]) == 1:
            if pd.isna(class_name) or class_name not in IMD_CLASSES:
                raise ValueError(
                    f"Row {row_index + 2}: cyclone images require an exact verified imd_class "
                    f"from {list(IMD_CLASSES)}."
                )
        elif not pd.isna(class_name) and str(class_name):
            raise ValueError(
                f"Row {row_index + 2}: non-cyclone records must not contain an intensity class."
            )

    present_classes = set(
        frame.loc[frame["cyclone_detected"] == 1, "imd_class"].astype(str)
    )
    missing_classes = [name for name in IMD_CLASSES if name not in present_classes]
    if missing_classes:
        raise ValueError(
            "The real labeled dataset must contain examples for all 8 IMD classes. "
            f"Missing: {missing_classes}. No labels are inferred automatically."
        )
    if set(frame["cyclone_detected"].unique()) != {0, 1}:
        raise ValueError("Training requires labeled cyclone and non-cyclone examples.")
    if frame["group_id"].nunique() < 5:
        raise ValueError(
            "At least five independent group_id values are required for train/validation/test splits."
        )

    frame["class_index"] = frame["imd_class"].map(
        {name: index for index, name in enumerate(IMD_CLASSES)}
    ).fillna(0).astype(np.int32)
    return frame


def split_by_group(
    frame: pd.DataFrame,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
        raise ValueError("Test and validation fractions must be positive and sum to less than 1.")

    groups = frame["group_id"].astype(str).to_numpy()
    all_indices = np.arange(len(frame))
    first_split = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    train_validation_indices, test_indices = next(first_split.split(all_indices, groups=groups))

    remaining_groups = groups[train_validation_indices]
    relative_val_fraction = validation_fraction / (1 - test_fraction)
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_val_fraction,
        random_state=random_state + 1,
    )
    train_relative, validation_relative = next(
        second_split.split(train_validation_indices, groups=remaining_groups)
    )
    split_indices = {
        "train": train_validation_indices[train_relative],
        "validation": train_validation_indices[validation_relative],
        "test": test_indices,
    }
    result = {name: frame.iloc[index].copy() for name, index in split_indices.items()}

    group_sets = {name: set(part["group_id"].astype(str)) for name, part in result.items()}
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise RuntimeError("Group leakage detected while splitting the dataset.")
    missing_training_classes = set(IMD_CLASSES) - set(
        result["train"].loc[result["train"]["cyclone_detected"] == 1, "imd_class"].astype(str)
    )
    if missing_training_classes:
        raise ValueError(
            "The group-based training split does not contain all eight IMD classes. "
            "Provide more independent cyclone groups per class; no random image-level split is used. "
            f"Missing from training: {sorted(missing_training_classes)}"
        )
    for split_name, part in result.items():
        if set(part["cyclone_detected"].unique()) != {0, 1}:
            raise ValueError(
                f"The {split_name} split must include cyclone and non-cyclone groups "
                "to measure/train the separate detection task."
            )
    return result


def build_model() -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(224, 224, 3), name="satellite_image")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="training_augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False
    features = backbone(augmentation(inputs), training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = tf.keras.layers.Dropout(0.25, name="head_dropout")(features)
    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "cyclone_detection": tf.keras.layers.Dense(
                2, activation="softmax", name="cyclone_detection"
            )(features),
            "intensity": tf.keras.layers.Dense(
                len(IMD_CLASSES), activation="softmax", name="intensity"
            )(features),
        },
        name="cyclo_sense_mobilenetv2",
    )
    return model


def _compile_model(model: Any, learning_rate: float) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "cyclone_detection": tf.keras.losses.SparseCategoricalCrossentropy(),
            "intensity": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"cyclone_detection": 1.0, "intensity": 1.0},
    )


def _make_dataset(frame: pd.DataFrame, batch_size: int, training: bool) -> Any:
    import tensorflow as tf

    image_paths = frame["image_path"].astype(str).to_numpy()
    detected = frame["cyclone_detected"].to_numpy(dtype=np.int32)
    class_indices = frame["class_index"].to_numpy(dtype=np.int32)
    intensity_weights = detected.astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, detected, class_indices, intensity_weights)
    )

    def load_example(path: Any, detection_label: Any, class_index: Any, intensity_weight: Any):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224], antialias=True)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return (
            image,
            {
                "cyclone_detection": detection_label,
                "intensity": class_index,
            },
            {
                "cyclone_detection": tf.constant(1.0, dtype=tf.float32),
                "intensity": intensity_weight,
            },
        )

    dataset = dataset.map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(frame), 4096), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _evaluate(model: Any, frame: pd.DataFrame, batch_size: int) -> dict[str, Any]:
    import tensorflow as tf

    inputs = tf.data.Dataset.from_tensor_slices(frame["image_path"].astype(str).to_numpy())

    def load_input(path: Any):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224], antialias=True)
        return tf.keras.applications.mobilenet_v2.preprocess_input(image)

    inputs = inputs.map(load_input, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    outputs = model.predict(inputs, verbose=0)
    detection_scores = outputs["cyclone_detection"]
    intensity_scores = outputs["intensity"]

    detection_true = frame["cyclone_detected"].to_numpy(dtype=np.int32)
    detection_pred = detection_scores.argmax(axis=1)
    cyclone_mask = detection_true == 1
    intensity_true = frame.loc[cyclone_mask, "class_index"].to_numpy(dtype=np.int32)
    intensity_pred = intensity_scores[cyclone_mask].argmax(axis=1)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        intensity_true,
        intensity_pred,
        labels=list(range(len(IMD_CLASSES))),
        average="macro",
        zero_division=0,
    )
    detection_precision, detection_recall, detection_f1, _ = precision_recall_fscore_support(
        detection_true,
        detection_pred,
        labels=[0, 1],
        average="binary",
        zero_division=0,
    )
    return {
        "test_rows": int(len(frame)),
        "cyclone_test_rows": int(cyclone_mask.sum()),
        "intensity_accuracy": float(accuracy_score(intensity_true, intensity_pred)),
        "intensity_macro_precision": float(macro_precision),
        "intensity_macro_recall": float(macro_recall),
        "intensity_macro_f1": float(macro_f1),
        "intensity_classification_report": classification_report(
            intensity_true,
            intensity_pred,
            labels=list(range(len(IMD_CLASSES))),
            target_names=list(IMD_CLASSES),
            output_dict=True,
            zero_division=0,
        ),
        "intensity_confusion_matrix": confusion_matrix(
            intensity_true, intensity_pred, labels=list(range(len(IMD_CLASSES)))
        ).tolist(),
        "detection_accuracy": float(accuracy_score(detection_true, detection_pred)),
        "detection_precision": float(detection_precision),
        "detection_recall": float(detection_recall),
        "detection_f1": float(detection_f1),
        "detection_confusion_matrix": confusion_matrix(
            detection_true, detection_pred, labels=[0, 1]
        ).tolist(),
    }


def _save_confusion_matrix(metrics: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 10))
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["intensity_confusion_matrix"]),
        display_labels=list(IMD_CLASSES),
    )
    display.plot(ax=axis, xticks_rotation=45, cmap="Blues", colorbar=False)
    axis.set_title("MobileNetV2 — IMD intensity test confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(
    manifest_path: str | Path = MANIFEST_PATH,
    epochs: int = 10,
    fine_tune_epochs: int = 0,
    batch_size: int = 16,
) -> dict[str, Any]:
    if epochs < 1 or fine_tune_epochs < 0 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive; fine_tune_epochs cannot be negative.")
    frame = load_manifest(manifest_path)
    splits = split_by_group(frame)
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required to train MobileNetV2. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    print(
        "Grouped data split: "
        + ", ".join(f"{name}={len(part)} rows/{part['group_id'].nunique()} groups" for name, part in splits.items())
    )
    model = build_model()
    _compile_model(model, learning_rate=1e-3)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
    ]
    model.fit(
        _make_dataset(splits["train"], batch_size, training=True),
        validation_data=_make_dataset(splits["validation"], batch_size, training=False),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if fine_tune_epochs:
        backbone = model.get_layer("mobilenetv2_1.00_224")
        backbone.trainable = True
        freeze_before = max(0, len(backbone.layers) - 30)
        for layer in backbone.layers[:freeze_before]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        _compile_model(model, learning_rate=1e-5)
        model.fit(
            _make_dataset(splits["train"], batch_size, training=True),
            validation_data=_make_dataset(splits["validation"], batch_size, training=False),
            epochs=fine_tune_epochs,
            callbacks=[
                callbacks[0],
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=4,
                    restore_best_weights=True,
                ),
            ],
            verbose=1,
        )

    best_model = tf.keras.models.load_model(MODEL_PATH)
    metrics = _evaluate(best_model, splits["test"], batch_size)
    metrics.update(
        {
            "architecture": "MobileNetV2 ImageNet backbone, frozen-head training with optional final-30-layer fine-tuning; separate binary detection and 8-class intensity softmax heads",
            "classes": list(IMD_CLASSES),
            "manifest_path": str(Path(manifest_path).resolve()),
            "split_group_counts": {
                name: int(part["group_id"].nunique()) for name, part in splits.items()
            },
            "split_row_counts": {name: int(len(part)) for name, part in splits.items()},
            "model_path": str(MODEL_PATH),
            "preprocessing": PREPROCESSING_CONFIG,
        }
    )
    (ROOT / "models" / "class_names.json").write_text(
        json.dumps(list(IMD_CLASSES), indent=2), encoding="utf-8"
    )
    (ROOT / "models" / "preprocessing.json").write_text(
        json.dumps(PREPROCESSING_CONFIG, indent=2), encoding="utf-8"
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_confusion_matrix(metrics, CONFUSION_MATRIX_PATH)

    print(
        "Test intensity metrics: "
        f"accuracy={metrics['intensity_accuracy']:.4f}, "
        f"precision={metrics['intensity_macro_precision']:.4f}, "
        f"recall={metrics['intensity_macro_recall']:.4f}, "
        f"F1={metrics['intensity_macro_f1']:.4f}"
    )
    print(
        "Test detection metrics: "
        f"accuracy={metrics['detection_accuracy']:.4f}, "
        f"precision={metrics['detection_precision']:.4f}, "
        f"recall={metrics['detection_recall']:.4f}, "
        f"F1={metrics['detection_f1']:.4f}"
    )
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 using a real, verified, group-labeled dataset."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    try:
        train(
            manifest_path=arguments.manifest,
            epochs=arguments.epochs,
            fine_tune_epochs=arguments.fine_tune_epochs,
            batch_size=arguments.batch_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Training stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
