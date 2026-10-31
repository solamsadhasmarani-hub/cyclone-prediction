from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from utils.prediction import (
    MODEL_PATH,
    ModelUnavailableError,
    load_model,
    predict_image,
)
from utils.preprocessing import SUPPORTED_EXTENSIONS, load_rgb_image
from utils.temporal_analysis import (
    derive_evolution_status,
    sort_observations,
    timestamp_from_filename,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGES_PER_REQUEST = 20


def _metadata_for_files() -> dict[str, dict[str, Any]]:
    raw = request.form.get("metadata")
    if not raw:
        return {}
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Metadata must be valid JSON.") from error

    if isinstance(metadata, list):
        result: dict[str, dict[str, Any]] = {}
        for row in metadata:
            if isinstance(row, dict) and row.get("filename"):
                result[str(row["filename"])] = row
        return result
    if isinstance(metadata, dict):
        return {
            str(filename): value
            for filename, value in metadata.items()
            if isinstance(value, dict)
        }
    raise ValueError("Metadata must be a JSON list or filename-keyed object.")


def _file_error(upload: Any) -> str | None:
    filename = upload.filename or ""
    if not filename:
        return "A filename is required."
    if Path(filename).suffix.casefold() not in SUPPORTED_EXTENSIONS:
        return "Supported image formats are JPG, JPEG, and PNG."
    return None


def _analyze_upload(upload: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    filename = upload.filename or ""
    extension_error = _file_error(upload)
    if extension_error:
        return {"filename": filename, "status": "error", "error": extension_error}

    image_bytes = upload.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return {
            "filename": filename,
            "status": "error",
            "error": "Each image must be 20 MB or smaller.",
        }
    if not image_bytes:
        return {"filename": filename, "status": "error", "error": "The uploaded image is empty."}

    try:
        load_rgb_image(image_bytes)
    except ValueError as error:
        return {"filename": filename, "status": "error", "error": str(error)}

    try:
        result = predict_image(image_bytes)
    except ValueError as error:
        return {"filename": filename, "status": "error", "error": str(error)}

    timestamp = metadata.get("timestamp") or timestamp_from_filename(filename)
    result.update(
        {
            "filename": filename,
            "timestamp": timestamp,
            "cyclone_id": metadata.get("cyclone_id") or metadata.get("group_id"),
            "latitude": metadata.get("latitude"),
            "longitude": metadata.get("longitude"),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    return result


@app.get("/health")
def health() -> Any:
    model_ready = False
    reason = "Model not trained yet. Run train_model.py."
    if MODEL_PATH.is_file():
        try:
            load_model()
            model_ready = True
            reason = "Trained MobileNetV2 model loaded."
        except ModelUnavailableError as error:
            reason = str(error)
    return jsonify(
        {
            "status": "ok",
            "model_ready": model_ready,
            "model_path": str(MODEL_PATH),
            "message": reason,
        }
    )


@app.post("/predict")
def predict() -> Any:
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"error": "Upload one image using the 'file' form field."}), 400
    try:
        metadata = _metadata_for_files().get(upload.filename or "", {})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    try:
        result = _analyze_upload(upload, metadata)
    except ModelUnavailableError as error:
        return jsonify({"error": str(error)}), 503
    if result["status"] == "error":
        return jsonify(result), 400
    return jsonify(result), 200


@app.post("/predict/multiple")
def predict_multiple() -> Any:
    uploads = request.files.getlist("files")
    if not uploads:
        uploads = request.files.getlist("file")
    if not uploads or all(not upload.filename for upload in uploads):
        return jsonify({"error": "Upload one or more images using the 'files' form field."}), 400
    if len(uploads) > MAX_IMAGES_PER_REQUEST:
        return jsonify({"error": f"A maximum of {MAX_IMAGES_PER_REQUEST} images is allowed."}), 413
    try:
        metadata_by_filename = _metadata_for_files()
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    results = []
    for upload in uploads:
        try:
            results.append(
                _analyze_upload(
                    upload,
                    metadata_by_filename.get(upload.filename or "", {}),
                )
            )
        except ModelUnavailableError as error:
            return jsonify({"error": str(error)}), 503

    valid_observations = [
        result for result in results
        if result.get("status") != "error"
        and result.get("cyclone_detected") is True
        and result.get("predicted_class")
    ]
    results = sort_observations(results)
    valid_observations = sort_observations(valid_observations)
    return jsonify(
        {
            "observations": results,
            "evolution_status": derive_evolution_status(valid_observations),
            "data_mode": "REAL MODEL INFERENCE",
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
