from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the backend service")
    app: str = Field(..., description="Application name")
    environment: str = Field(..., description="Runtime environment")


class AnalyzeImageRequest(BaseModel):
    image_path: str | None = Field(default=None, description="Optional local path for a demo image")


class AnalyzeImageResponse(BaseModel):
    cyclone_detected: bool
    detection_confidence: float
    classification: str | None = None
    classification_confidence: float | None = None
    explanation_image: str | None = None
    status: str = "success"
