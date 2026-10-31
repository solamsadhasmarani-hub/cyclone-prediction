from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from ai.inference import (
    PrototypeInferenceUnavailableError,
    get_model_status,
    predict_uploaded_image,
)
from backend.app.config import settings
from backend.app.services.dashboard import build_demo_dashboard

app = FastAPI(title="SIH26070 Cyclone AI Monitor", version="0.1.0")
cors_origins = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeImageResponse(BaseModel):
    cyclone_detected: bool
    detection_confidence: float
    classification: str | None = None
    classification_confidence: float | None = None
    stage_probabilities: dict[str, float] | None = None
    imd_classification: str | None = None
    imd_confidence: float | None = None
    imd_probabilities: dict[str, float] | None = None
    analysis_timestamp: str
    latitude: float | None = None
    longitude: float | None = None
    wind_speed_kmh: float | None = None
    evolution_status: str | None = None
    model_architecture: str
    model_provenance: str
    data_mode: str
    explanation_image: str | None = None
    status: str = "success"


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return build_demo_dashboard()["health"]  # type: ignore[return-value]


@app.get("/api/models/status")
def model_status() -> dict[str, bool | str]:
    return get_model_status()


@app.get("/api/dashboard")
def dashboard() -> dict[str, object]:
    payload = build_demo_dashboard()
    status = get_model_status()
    payload["model_status"] = status
    payload["model_capabilities"] = {
        "uploaded_image_detection": status["detection_model_ready"],
        "four_stage_prototype_classification": status["classification_model_ready"],
        "imd_intensity_estimation": False,
        "track_forecasting": False,
        "gradcam": status["gradcam_ready"],
        "risk_assessment": False,
    }
    payload["model_information"]["available_prototype_architecture"] = status["model_architecture"]
    payload["model_information"]["model_provenance"] = status["model_provenance"]
    return payload


@app.post("/api/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image(file: UploadFile = File(...)) -> AnalyzeImageResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Image file is required")

    file_bytes = await file.read(10 * 1024 * 1024 + 1)
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller")

    try:
        with Image.open(BytesIO(file_bytes)) as uploaded_image:
            uploaded_image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from error

    try:
        result = predict_uploaded_image(file_bytes)
    except PrototypeInferenceUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    result["analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
    return AnalyzeImageResponse(**result)


@app.get("/api/cyclones")
def list_cyclones() -> list[dict[str, str]]:
    dashboard_data = build_demo_dashboard()
    return dashboard_data["cyclones"]  # type: ignore[return-value]


@app.get("/api/cyclones/{cyclone_id}")
def get_cyclone(cyclone_id: str) -> dict[str, str]:
    return {"cyclone_id": cyclone_id, "cyclone_name": "Demo Cyclone", "status": "demo"}


@app.get("/api/cyclones/{cyclone_id}/track")
def get_track(cyclone_id: str) -> dict[str, object]:
    return {
        "cyclone_id": cyclone_id,
        "data_mode": "DEMO DATA — NOT A MODEL FORECAST",
        "predicted_track": [{"hours": 12, "latitude": 15.4, "longitude": 82.8}],
        "historical_track": [{"hours": 0, "latitude": 12.2, "longitude": 80.8}],
    }


@app.get("/api/cyclones/{cyclone_id}/intensity")
def get_intensity(cyclone_id: str) -> dict[str, object]:
    return {
        "cyclone_id": cyclone_id,
        "data_mode": "DEMO DATA — NOT A MODEL FORECAST",
        "current_intensity": 98,
        "predicted_intensity": 108,
        "prediction_horizon_hours": 12,
    }


@app.get("/api/predictions")
def list_predictions() -> list[dict[str, object]]:
    dashboard_data = build_demo_dashboard()
    return dashboard_data["predictions"]  # type: ignore[return-value]


@app.get("/api/predictions/summary")
def prediction_summary() -> list[dict[str, object]]:
    return list_predictions()


@app.get("/api/healthz")
def health_check_alias() -> dict[str, str]:
    return health_check()
