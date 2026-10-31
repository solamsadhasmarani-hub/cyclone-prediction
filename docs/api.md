# API Design

The API layer is implemented in Phase 7 using FastAPI and Pydantic.

Implemented demo endpoints:
- GET /api/health
- GET /api/models/status
- GET /api/dashboard
- POST /api/analyze-image
- GET /api/cyclones
- GET /api/cyclones/{cyclone_id}
- GET /api/cyclones/{cyclone_id}/track
- GET /api/cyclones/{cyclone_id}/intensity
- GET /api/predictions
- GET /api/predictions/summary
- GET /api/healthz

`GET /api/dashboard` returns a centralized demo payload with eight IMD class probability examples, timestamped sample observations, a derived sample evolution state, separately labeled observed and predicted tracks, and model-profile metadata. Dashboard probabilities and tracks are explicitly marked as demo data.

`POST /api/analyze-image` accepts an image up to 10 MB and returns available ResNet-18 prototype detection and four-stage classification fields. It does not return an eight-class IMD classification, wind speed, coordinates, or temporal evolution; those fields remain unavailable rather than being inferred. The existing classifier was trained on synthetic data. The proposed MobileNetV2 eight-class model is not yet implemented.

The PostgreSQL layer introduced in Phase 8 defines the database model structure, but the dashboard payload currently uses centralized demo data rather than database observations.

These endpoints are intended for local development and prototype validation, not operational forecasting or warnings.
