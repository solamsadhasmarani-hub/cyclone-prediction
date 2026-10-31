from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.app.main import app


def validate_dashboard_contract() -> None:
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200, f"health endpoint failed: {health.status_code}"
    assert health.json()["status"] == "ok", "health endpoint did not report healthy state"

    model_status = client.get("/api/models/status")
    assert model_status.status_code == 200, f"model status failed: {model_status.status_code}"
    assert model_status.json()["demo_mode"] is True, "demo mode should be enabled"

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200, f"dashboard endpoint failed: {dashboard.status_code}"
    dashboard_payload = dashboard.json()
    assert dashboard_payload["health"]["status"] == "ok", "dashboard health payload invalid"
    assert len(dashboard_payload["cyclones"]) >= 2, "dashboard should include at least two cyclone records"
    assert len(dashboard_payload["intensity_forecast"]) >= 4, "intensity forecast payload is incomplete"
    assert len(dashboard_payload["track_prediction"]) >= 3, "track prediction payload is incomplete"
    assert len(dashboard_payload["predictions"]) >= 2, "prediction payload is incomplete"

    prediction_summary = client.get("/api/predictions/summary")
    assert prediction_summary.status_code == 200, f"prediction summary failed: {prediction_summary.status_code}"
    assert prediction_summary.json() == dashboard_payload["predictions"], "prediction summary does not match dashboard payload"

    cyclone_response = client.get("/api/cyclones/DEMO_01")
    assert cyclone_response.status_code == 200, f"cyclone route failed: {cyclone_response.status_code}"
    assert cyclone_response.json()["cyclone_id"] == "DEMO_01", "cyclone route returned unexpected payload"

    print("Phase 11 validation passed: API contract and dashboard integration are healthy.")


if __name__ == "__main__":
    validate_dashboard_contract()
