from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_frontend_cors_allows_local_vite_origin():
    response = client.options(
        "/api/dashboard",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_models_status_endpoint():
    response = client.get("/api/models/status")
    assert response.status_code == 200
    assert response.json()["demo_mode"] is True


def test_analyze_image_without_file_fails():
    response = client.post("/api/analyze-image")
    assert response.status_code == 422


def test_cyclone_route():
    response = client.get("/api/cyclones/DEMO_01")
    assert response.status_code == 200
    assert response.json()["cyclone_id"] == "DEMO_01"
