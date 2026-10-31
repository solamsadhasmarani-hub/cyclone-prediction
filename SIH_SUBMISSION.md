# SIH26070 Submission Summary

## Project Overview

**Smart India Hackathon 2026: CYCLO SENSE — AI/ML-based Tropical Cyclone Pattern Identification, Classification, and Prediction**

SIH26070 is a demo-safe cyclone analysis prototype with satellite-image upload, a FastAPI API, and an integrated React dashboard. Dashboard observations, probabilities, tracks, and alerts are synthetic demo records; the project is not a live or authoritative forecast system.

## Proposed classification design

The SIH solution proposes **MobileNetV2 + transfer learning + a custom classification head**, using 224 × 224 satellite-image inputs and exactly these eight primary intensity classes:

1. Low Pressure Area
2. Depression
3. Deep Depression
4. Cyclonic Storm
5. Severe Cyclonic Storm
6. Very Severe Cyclonic Storm
7. Extremely Severe Storm
8. Super Cyclonic Storm

This eight-class MobileNetV2 model is **not implemented in the current prototype**. The eight-class dashboard probability distribution is explicitly illustrative demo data, not model output.

## Key Features

### Data Management
- Synthetic cyclone image dataset with metadata validation and preprocessing
- Leakage-safe train/validation/test splitting by cyclone ID to prevent data contamination
- Local sample dataset generation for reproducible demo workflows

### AI Models

#### Detection (Phase 2)
- ResNet-18 prototype model for cyclone presence detection
- Binary classification with confidence scoring
- Synthetic demo training pipeline

#### Classification (Phase 3)
- Multi-class cyclone stage classification (developing, mature, weakening, dissipating)
- ResNet-18-based with class confusion matrix and precision/recall reporting
- Demo evaluation on synthetic cyclone imagery
- These four prototype stages are not the eight primary IMD intensity classes

#### Explainability (Phase 4)
- Grad-CAM-style heatmap generation for model decision visualization
- Explanation metadata and confidence attribution
- Fallback image generation for robust demo validation

#### Intensity Forecasting (Phase 5)
- Random Forest regression model for cyclone wind-speed prediction
- Synthetic intensity data with future forecasting targets
- MAE and RMSE evaluation metrics

#### Track Prediction (Phase 6)
- Random Forest model for latitude/longitude trajectory forecasting
- Geospatial track point generation and RMSE validation
- 12-24 hour prediction horizon

### Backend API (Phase 7-10)
- FastAPI-based REST service with CORS support
- `/api/health` health check endpoint
- `/api/dashboard` unified demo payload endpoint
- `/api/models/status` model readiness status
- `/api/predictions/summary` prediction results
- `/api/analyze-image` cyclone detection and classification endpoint
- Image file upload for prototype inference; reports only fields available from the current model
- The upload API does not return an eight-class IMD result, wind speed, location, or temporal evolution

### Database Schema (Phase 8)
- SQLAlchemy ORM models for:
  - Cyclones (metadata, status, location)
  - Satellite Images (timestamps, file references)
  - Observations (intensity, wind speed, pressure)
  - Predictions (detection, classification, track, intensity)
  - Track Points (geospatial trajectory)
  - Model Results (performance metadata)
- Local SQLite database for development
- Ready for PostgreSQL migration in production

### Frontend Dashboard (Phase 9-10)
- React + Vite responsive dashboard
- Professional dark-mode UI with gradients and card-based layout
- Real-time data binding to backend API
- Sections:
  - System health and persistent DEMO MODE label
  - Cyclone detection, primary IMD demo classification, confidence, and supporting wind speed
  - Eight-class probability chart explicitly labeled DEMO DATA
  - Timestamped demo intensity observations and trend-derived evolution label
  - Observed track and predicted track shown separately
  - Future numerical wind-speed forecast clearly separated from current classification
  - Satellite upload and available prototype inference result fields
  - Historical demo records, alerts, and honest model-performance status

### Integration & Validation (Phase 11)
- Smoke test validation for API contract consistency
- Full Python pytest suite (20 tests) covering all modules
- Dashboard payload validation, including class probabilities, timestamps, track separation, and evolution derivation
- Frontend build verification with Vite

### Deployment & Demo (Phase 12)
- Single-command local launch: `python scripts/start_demo.py`
- Simultaneous backend (Uvicorn) and frontend (Vite) startup
- Deployment guide at `docs/deployment.md`
- Demo-safe: all data is synthetic and clearly marked for showcase-only use

## Technology Stack

**Backend:**
- FastAPI 0.104+
- SQLAlchemy 2.0+
- Uvicorn
- PyTorch for model inference

**Frontend:**
- React 18+
- Vite 5.4+
- Recharts for charting
- CSS Grid for responsive layout

**AI/ML:**
- PyTorch (current prototype image models use ResNet-18)
- Proposed, not-yet-implemented image classifier: MobileNetV2 with transfer learning and eight IMD outputs
- Scikit-learn (Random Forest regressors)
- NumPy, PIL for data processing

**Testing & Validation:**
- pytest for Python test suite
- FastAPI TestClient for API validation
- Vite for frontend build validation

## Project Structure

```
SIH26070/
├── backend/
│   ├── app/
│   │   ├── models/database_models.py
│   │   ├── schemas/api.py
│   │   ├── services/dashboard.py
│   │   ├── main.py
│   │   └── config.py
│   ├── requirements.txt
│   └── .env.example
├── ai/
│   ├── detection/     (ResNet-18 cyclone detector)
│   ├── classification/ (Multi-class cyclone stage classifier)
│   ├── explainability/ (Grad-CAM heatmap generation)
│   ├── intensity/      (Wind speed prediction model)
│   └── tracking/       (Cyclone trajectory predictor)
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── services/api.js
│   │   ├── components/StatCard.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
├── data/              (Synthetic demo dataset)
├── models/            (Trained model checkpoints)
├── scripts/
│   ├── phase1_demo.py
│   ├── phase11_validation.py
│   └── start_demo.py
├── tests/             (18 validation tests across phases)
├── docs/
│   ├── deployment.md
│   ├── architecture.md
│   ├── api.md
│   ├── model.md
│   └── database.md
└── README.md
```

## Running the Demo

### Prerequisites
- Python 3.10+
- Node.js 18+
- Virtual environment activated: `.venv\Scripts\activate`

### Launch

```bash
python scripts/start_demo.py
```

This starts:
- **Backend API**: http://localhost:8000
- **Frontend Dashboard**: http://localhost:5173

### Validation

```bash
# Run smoke check
.venv\Scripts\python scripts/phase11_validation.py

# Run full test suite
.venv\Scripts\python -m pytest -q

# Build frontend for production validation
npm --prefix frontend run build
```

## Design Principles

1. **Demo-Safe**: All data is synthetic and local. No real cyclone warnings or authoritative meteorological claims.
2. **Modular**: Each phase (1–12) stands independently but composes into a complete system.
3. **Leakage-Free**: Data splits by cyclone ID, not random rows, to prevent train/test contamination.
4. **Accurate model status**: Separates the proposed MobileNetV2 eight-class architecture from the implemented synthetic ResNet-18 baseline.
5. **Integrated Flow**: Backend, database, and frontend are connected end-to-end with a unified demo payload.

## What This Is NOT

- Not an authoritative cyclone forecasting system
- Not connected to real satellite APIs or meteorological data sources
- Not a production warning system; strictly for demonstration and evaluation
- Dashboard IMD probabilities, observations, track coordinates, and alerts are synthetic demo values
- Does not claim trained-model accuracy/precision/recall on the dashboard; those metrics are not exposed by the integrated API
- Not claiming to replace official IMD, NOAA, or other operational cyclone services

## Future Extensions

- PostgreSQL database backend for production deployment
- Real satellite image ingestion (NOAA, INSAT APIs)
- Live geospatial mapping with Leaflet/Mapbox
- Stronger deep-learning baselines (LSTM, Transformers for sequences)
- Multi-source data fusion (radar, atmospheric pressure, sea-surface temperature)
- User authentication and role-based access control
- Alert notification system for critical predictions

## Team & Attribution

Built for Smart India Hackathon 2026 as a demonstration of AI/ML capabilities in climate and disaster management.

**Project Type**: Demo-safe AI prototype  
**Status**: Phase 12 (Deployment Ready)  
**Last Updated**: October 1, 2026
