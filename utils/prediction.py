from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np

from ai.config.imd_classes import IMD_CLASSES
from utils.preprocessing import PREPROCESSING_CONFIG, image_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "cyclone_mobilenetv2.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"
PREPROCESSING_PATH = PROJECT_ROOT / "models" / "preprocessing.json"
DETECTION_THRESHOLD = 0.5


class ModelUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model(model_path: str = str(MODEL_PATH)) -> Any:
    path = Path(model_path)
    if not path.is_file():
        raise ModelUnavailableError("Model not trained yet. Run train_model.py.")
    if not CLASS_NAMES_PATH.is_file() or not PREPROCESSING_PATH.is_file():
        raise ModelUnavailableError(
            "Model metadata is missing. Retrain the model with train_model.py."
        )

    try:
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        preprocessing = json.loads(PREPROCESSING_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelUnavailableError(f"Could not read model metadata: {error}") from error

    if class_names != list(IMD_CLASSES):
        raise ModelUnavailableError(
            "Saved class names do not match the required eight IMD classes. Retrain the model."
        )
    if preprocessing != PREPROCESSING_CONFIG:
        raise ModelUnavailableError(
            "Saved preprocessing configuration does not match this application. Retrain the model."
        )

    try:
        from tensorflow.keras.models import load_model as keras_load_model

        model = keras_load_model(path)
    except (ImportError, OSError, ValueError) as error:
        raise ModelUnavailableError(f"Could not load the trained MobileNetV2 model: {error}") from error
    return model


def predict_image(source: bytes | bytearray | str | Path) -> dict[str, Any]:
    model = load_model()
    inputs = image_tensor(source)
    raw_outputs = model.predict(inputs, verbose=0)

    if isinstance(raw_outputs, dict):
        detection_scores = raw_outputs["cyclone_detection"][0]
        intensity_scores = raw_outputs["intensity"][0]
    elif isinstance(raw_outputs, (list, tuple)) and len(raw_outputs) == 2:
        detection_scores, intensity_scores = (output[0] for output in raw_outputs)
    else:
        raise ModelUnavailableError("Model output does not contain detection and intensity heads.")

    detection_scores = np.asarray(detection_scores, dtype=float)
    intensity_scores = np.asarray(intensity_scores, dtype=float)
    if detection_scores.shape != (2,) or intensity_scores.shape != (len(IMD_CLASSES),):
        raise ModelUnavailableError("Model output dimensions do not match the expected classes.")
    if not np.isfinite(detection_scores).all() or not np.isfinite(intensity_scores).all():
        raise ModelUnavailableError("Model returned non-finite prediction values.")

    detected = int(np.argmax(detection_scores)) == 1
    result: dict[str, Any] = {
        "cyclone_detected": detected,
        "detection_confidence": float(detection_scores[1 if detected else 0]),
        "predicted_class": None,
        "class_index": None,
        "confidence": None,
        "probabilities": None,
        "model_architecture": "MobileNetV2 multitask transfer learning",
        "data_mode": "REAL MODEL INFERENCE",
    }
    if not detected:
        result["status"] = "Non-Cyclone Image"
        return result

    class_index = int(np.argmax(intensity_scores))
    result.update(
        {
            "predicted_class": IMD_CLASSES[class_index],
            "class_index": class_index,
            "confidence": float(intensity_scores[class_index]),
            "probabilities": {
                name: float(intensity_scores[index])
                for index, name in enumerate(IMD_CLASSES)
            },
            "top_3": [
                {
                    "class_name": IMD_CLASSES[index],
                    "class_index": int(index),
                    "probability": float(intensity_scores[index]),
                }
                for index in np.argsort(intensity_scores)[-3:][::-1]
            ],
            "status": "success",
        }
    )
    return result
