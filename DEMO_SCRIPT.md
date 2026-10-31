# CYCLO SENSE — SIH26070 Demo Script

Use this guide to showcase the cyclone AI system during the SIH presentation.

## Opening Statement

> "Cyclo Sense is our SIH26070 cyclone-analysis prototype. It brings image detection, intensity classification concepts, timestamped analysis, track views, and forecasts into one dashboard. This running demo uses synthetic data and is not a real warning system. The proposed MobileNetV2 eight-class classifier is distinguished from the current ResNet-18 prototype."

## Demo Flow (5–10 minutes)

### 1. Launch the Stack (1 min)
```bash
# Terminal
python scripts/start_demo.py
```

Wait for:
```
  ➜  Local:   http://localhost:5173/
  ...
  Uvicorn running on http://0.0.0.0:8000
```

### 2. Open Dashboard (30 sec)
- Navigate to **http://localhost:5173** in the browser
- Point out:
  - **Sidebar**: Project branding and navigation
  - **Top bar**: "CYCLO SENSE" with healthy system and DEMO MODE badges
  - **Hero card**: Cyclone detection, eight-class IMD demo classification, confidence, and supporting wind speed

### 3. Highlight Key Metrics (1 min)
Scroll down and show:
- **Active cyclone count**: 2 (synthetic demo records)
- **Latest analysis**: 16:00 UTC (demo timestamp)
- **Detection status**: YES; 94% illustrative confidence
- **IMD intensity classification**: Severe Cyclonic Storm (demo)
- **Wind speed**: 98 km/h as supporting demo information

Emphasize that the IMD probability chart, observations, and coordinates are demo data, not satellite-derived or official observations.

### 4. Show Intensity Forecast Chart (1 min)
- Point to the area chart showing wind-speed prediction over 30 hours
- Explain: "This prototype forecast chart is separate from the primary IMD intensity classification."
- Values are synthetic demo forecast values, not an operational prediction.

### 5. Display Track Prediction (1 min)
- Show **Observed track** separately from **Predicted track**.
- Observed coordinates have timestamps; predicted coordinates are future demo samples.

### 6. Predictions Summary Table (30 sec)
- Show the current analysis summary and the derived evolution label.
- Explain that evolution comes from the ordered sample intensity observations.

### 7. Show the Upload Workflow (optional, 1 min)
- Open Satellite Upload and choose one of the local sample images.
- The endpoint runs the existing ResNet-18 prototype detector and its four-stage classifier when detection is positive.
- Explain that the current image model does not return an eight-class IMD prediction, wind speed, location, or temporal evolution. Those fields remain unavailable rather than being guessed.

### 8. Open API Docs (optional, 1 min)
- Go to **http://localhost:8000/docs** (FastAPI auto-generated docs)
- Show endpoints:
  - `/api/health`
  - `/api/dashboard`
  - `/api/models/status`
  - `/api/predictions/summary`

### 9. Explain the Architecture (2 min)
Use these key points:

**Data Pipeline:**
- Synthetic satellite images preprocessed and validated
- Leakage-safe splits by cyclone ID

**AI Models:**
1. **Proposed SIH classifier**: MobileNetV2 + transfer learning + custom head; 224 × 224 input and eight IMD classes. This is a proposal, not the current implementation.
2. **Current image prototype**: ResNet-18 detector and ResNet-18 four-stage classifier trained on synthetic data.
3. **Other baselines**: Random Forest models are present for intensity and track experiments; dashboard sample forecasts are explicitly demo data.
4. **Evaluation**: The dashboard does not publish metrics because the integrated API currently does not provide evaluated metrics.

**Backend:**
- FastAPI REST service with CORS for frontend integration
- SQLAlchemy database schema ready for PostgreSQL

**Frontend:**
- React + Vite dashboard with Recharts charts
- Real-time data binding to backend
- Responsive dark-mode UI

### 10. Show Code Highlights (optional, 1 min)
If asked about implementation:
- Show [backend/app/main.py](../../backend/app/main.py) — simple FastAPI endpoints
- Show [frontend/src/App.jsx](../../frontend/src/App.jsx) — React hooks and data fetching
- Show [ai/detection/model.py](../../ai/detection/model.py) — ResNet-18 model definition

### 11. Validation (optional, 1 min)
If time permits, show:
```bash
# Terminal
.venv\Scripts\python scripts\phase11_validation.py
# Output: "Phase 11 validation passed: API contract and dashboard integration are healthy."
```

## Key Talking Points

### Problem Statement
- Tropical cyclones are complex weather phenomena requiring rapid analysis
- Manual satellite image review is time-consuming; AI can assist in real-time detection and forecasting
- Multiple AI tasks (detection, classification, intensity, track) benefit from a unified workflow

### Solution
- Proposed multi-stage pipeline: preprocess → detect/classify → confidence/probability → temporal evolution/track view
- Modular design: each model can be improved independently
- Integrated dashboard: unified interface for multiple predictions
- Demo-safe: all data synthetic, suitable for hackathon showcase

### Why This Approach?
- **Clear implementation status**: proposed MobileNetV2 eight-class classifier is separated from the implemented ResNet-18 synthetic baseline
- **Leakage-free data**: Splitting by cyclone ID ensures realistic evaluation
- **End-to-end integration**: Not just models, but a working UI/API system
- **Deployable**: Uses FastAPI + React stack, ready for Docker/cloud deployment

### Demo-Safe Assurances
> "This uses 100% synthetic data generated locally. We're not claiming to replace IMD or NOAA; we're demonstrating how AI workflows can be built and evaluated for such use cases."

## Troubleshooting During Demo

| Issue | Solution |
|-------|----------|
| Dashboard does not load | Check the terminal where `start_demo.py` was launched and refresh after both services start |
| Dashboard API connection error | Backend is not running; restart `start_demo.py` |
| Charts or track table not rendering | Browser cache; press Ctrl+Shift+R (hard refresh) |
| API endpoints show 404 | Make sure backend is on http://localhost:8000, not 5173 |
| Frontend won't start | Ensure Node.js/npm installed; run `npm --prefix frontend install` |

## Questions You Might Get

**Q: Are these live observations or official cyclone warnings?**  
A: No. Dashboard probabilities, timestamps, tracks, and alerts are synthetic demonstration values. The uploaded-image endpoint only returns fields from the current prototype models.

**Q: Can this replace real cyclone warnings?**  
A: No. This is a research/demo prototype. Real systems need validation against official meteorological data, regulatory approval, and operational integration.

**Q: Does the current image classifier implement the eight IMD classes?**  
A: No. The eight-class MobileNetV2 pipeline is the proposed design. The current image classification baseline uses four synthetic development stages with ResNet-18.

**Q: Can we integrate real satellite data?**  
A: A validated data ingestion and model retraining workflow would be required; the current project has no live satellite integration.

## Post-Demo Follow-Up

- Provide [SIH_SUBMISSION.md](SIH_SUBMISSION.md) summarizing the full project
- Share [docs/deployment.md](../docs/deployment.md) for local setup
- Link to the GitHub repository (if public)
- Offer to run any validation tests on demand

---

**Demo Duration**: 5–10 minutes  
**Difficulty**: Beginner-friendly (suitable for all audiences)  
**Setup Time**: ~2 minutes (if already built)
