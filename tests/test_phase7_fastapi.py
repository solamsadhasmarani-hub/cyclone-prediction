from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_frontend_cors_allows_local_vite_origin():
    for origin in ("http://localhost:5173", "http://127.0.0.1:4173"):
        response = client.options(
            "/api/dashboard",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers["access-control-allow-origin"] == origin


def test_models_status_endpoint():
    response = client.get("/api/models/status")
    assert response.status_code == 200
    assert response.json()["demo_mode"] is True
    assert response.json()["intensity_model_ready"] is False
    assert response.json()["track_model_ready"] is False
    assert response.json()["risk_model_ready"] is False
    assert response.json()["model_provenance"].startswith("These prototype checkpoints")


def test_analyze_image_without_file_fails():
    response = client.post("/api/analyze-image")
    assert response.status_code == 422


def test_analyze_image_runs_real_checkpoint_and_returns_supported_fields():
    image = BytesIO()
    Image.new("RGB", (64, 48), color=(25, 45, 80)).save(image, format="PNG")

    response = client.post(
        "/api/analyze-image",
        files={"file": ("sample.png", image.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result["cyclone_detected"], bool)
    assert 0.0 <= result["detection_confidence"] <= 1.0
    assert result["data_mode"] == "SYNTHETIC-TRAINED PROTOTYPE INFERENCE"
    assert "not been validated on real satellite imagery" in result["model_provenance"]
    assert result["imd_classification"] is None
    if result["cyclone_detected"]:
        assert result["classification"] in {"developing", "mature", "weakening", "dissipating"}
        assert result["stage_probabilities"] is not None
        assert result["explanation_image"].startswith("data:image/png;base64,")
    else:
        assert result["classification"] is None
        assert result["stage_probabilities"] is None
        assert result["explanation_image"] is None


def test_cyclone_route():
    response = client.get("/api/cyclones/DEMO_01")
    assert response.status_code == 200
    assert response.json()["cyclone_id"] == "DEMO_01"
