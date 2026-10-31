from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.dashboard import derive_evolution_status


client = TestClient(app)


def test_dashboard_endpoint_has_integrated_payload():
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["health"]["status"] == "ok"
    assert data["model_status"]["demo_mode"] is True
    assert data["selected_cyclone"]["imd_classification"] == "Severe Cyclonic Storm"
    assert data["probabilities_source"] == "demo"
    assert len(data["cyclones"]) >= 2
    assert len(data["intensity_forecast"]) >= 4
    assert len(data["imd_intensity_probabilities"]) == 8
    assert sum(item["probability"] for item in data["imd_intensity_probabilities"]) == 100
    assert len(data["timestamped_observations"]) >= 2
    assert data["evolution_status"] == "STRENGTHENING"
    assert all("timestamp" in point for point in data["observed_track"])
    assert all("hours" in point for point in data["predicted_track"])
    assert data["model_information"]["proposed_architecture"].startswith("MobileNetV2")
    assert data["model_information"]["current_prototype_architecture"] == "ResNet-18"
    assert data["model_metrics"] is None
    assert len(data["track_prediction"]) >= 3
    assert len(data["predictions"]) >= 2


def test_prediction_summary_matches_dashboard():
    dashboard = client.get("/api/dashboard").json()
    summary = client.get("/api/predictions/summary").json()
    assert summary == dashboard["predictions"]


def test_invalid_satellite_upload_is_rejected():
    response = client.post(
        "/api/analyze-image",
        files={"file": ("not-an-image.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image"


def test_evolution_is_derived_from_timestamped_classes():
    observations = [
        {"timestamp": "2026-10-01T14:00:00Z", "classification": "Severe Cyclonic Storm"},
        {"timestamp": "2026-10-01T13:00:00Z", "classification": "Cyclonic Storm"},
    ]
    assert derive_evolution_status(observations) == "STRENGTHENING"
    assert derive_evolution_status(observations[:1]) == "INSUFFICIENT TEMPORAL DATA"

    observations[0]["classification"] = "Cyclonic Storm"
    assert derive_evolution_status(observations) == "PEAK"

    observations[0]["classification"] = "Depression"
    assert derive_evolution_status(observations) == "WEAKENING"
