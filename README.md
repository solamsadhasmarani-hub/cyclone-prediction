# CYCLO SENSE — SIH26070

This existing React/Vite and FastAPI project is the demo-safe prototype for Smart India Hackathon problem SIH26070. It uses synthetic sample data and does not provide operational cyclone warnings.

## Current phase

The repository is now in Phase 12: deployment and demo preparation.
- synthetic data pipeline and leakage-safe preprocessing
- detection and classification baselines
- explainability, intensity forecasting, and track prediction models
- FastAPI backend with demo endpoints and CORS-enabled frontend integration
- SQLAlchemy database schema foundation for cyclone records and predictions
- React dashboard shell wired to the backend demo data flow
- smoke validation covering API health, dashboard payloads, prediction summaries, and model status checks
- local launch script and deployment guide for a showcase-ready demo stack
- functional dashboard, satellite analysis, track, intensity classification, history, model performance, and alerts views
- eight IMD probability values and timestamped observations labeled as demo data
- an explicit distinction between the proposed MobileNetV2 eight-class model and the implemented synthetic ResNet-18 prototype

This state is intentionally demo-safe and local-first. It is a modular prototype for SIH showcase work, not an authoritative operational meteorological system.

## Standalone CYCLO SENSE model-training pipeline

The repository also contains a separate Flask API and Streamlit dashboard. These additions do not replace the existing React/Vite + FastAPI + PyTorch prototype. Both upload flows use the tracked PyTorch ResNet-18 detector and four-stage classifier checkpoints. Those checkpoints were trained on generated synthetic demo images and are not validated for real satellite imagery.

The repository does not include a trained eight-class IMD, intensity-forecasting, track-forecasting, or risk model. The real-label manifest currently has only two cyclone groups and five of the eight IMD classes, with no non-cyclone examples, so `train_model.py` correctly refuses to train the proposed MobileNetV2 model. Uploaded-image ResNet predictions are labeled as synthetic-trained prototype output, not scientific or operational predictions.

### Dataset manifest

Prepare `data/metadata/labels.csv` using the schema and validation requirements in [data/metadata/README.md](data/metadata/README.md). Every cyclone needs an explicit, verified IMD class, and each image must have a group identifier so images from the same cyclone/event cannot leak across splits. Non-cyclone examples need an explicit detection label and a blank `imd_class`.

Wind measurements are not automatically converted into IMD labels. Verify the source field, units, averaging interval, and authoritative category mapping before entering labels.

### Environment

Use a supported Python 3.11 environment for the separate MobileNetV2 training pipeline. TensorFlow versions newer than 2.10 do not support native Windows execution; on Windows, use WSL2/Ubuntu for TensorFlow training. The existing project `.venv` uses Python 3.14; its requirements intentionally omit TensorFlow on Windows. This limitation affects `train_model.py` and `predict.py`, not the existing PyTorch ResNet-18 upload dashboards.

To run the React/Vite dashboard with its FastAPI API:

```powershell
cd "C:\Users\solam\Documents\c++ programs\SIH26070"
.\.venv\Scripts\python.exe scripts\start_demo.py
```

For the separate Streamlit dashboard, start Flask in one terminal and Streamlit in another:

```powershell
.\.venv\Scripts\python.exe -m backend.api
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

From the repository root in a WSL2/Ubuntu terminal:

```bash
cd "/mnt/c/Users/solam/Documents/c++ programs/SIH26070"
python3.11 -m venv .venv-cyclo
source .venv-cyclo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Train the model

Training requires verified real images and the manifest above. It stops if the manifest, either detection class, any of the eight IMD classes, or sufficient independent storm groups are missing. The train/validation/test split is grouped by `group_id`.

```bash
python train_model.py --epochs 10
```

Optional fine-tuning of the final MobileNetV2 layers:

```bash
python train_model.py --epochs 10 --fine-tune-epochs 5
```

The best model is saved to `models/cyclone_mobilenetv2.keras`; class and preprocessing metadata are saved alongside it. Test metrics and the confusion matrix are written to `outputs/metrics.json` and `outputs/charts/confusion_matrix.png`. No training is triggered when either server starts.

### Start the Flask API

In one terminal with the WSL environment activated:

```bash
python -m backend.api
```

The API listens on `http://127.0.0.1:5000`. Check `GET /health` for the available ResNet-18 prototype checkpoints and their synthetic-training provenance.

### Start the Streamlit dashboard

In another terminal with the same WSL environment activated:

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`). Upload JPG/JPEG/PNG files on **Analyze Satellite Image**. Optional metadata CSV columns are `filename`, `timestamp`, `cyclone_id`, `latitude`, and `longitude`. Filenames with an unambiguous date/time may supply a timestamp when the CSV does not.

### Direct prediction

After a successful training run:

```bash
python predict.py "data/raw/hursat/example.png"
```

For a single prediction, send a multipart form field named `file` to `POST /predict`. For multiple images, send repeated multipart fields named `files` to `POST /predict/multiple`. Optional metadata can be sent as JSON in a form field named `metadata`.

Images rejected by the detection head are reported as **Non-Cyclone Image** and do not receive an intensity classification. Temporal trends require timestamps, and observed track maps require valid latitude/longitude metadata. The dashboard never substitutes demo probabilities when the backend or trained model is unavailable.

## Folder structure

```text
SIH26070/
├── backend/
│   ├── app/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── routes/
│   │   ├── services/
│   │   └── utils/
│   ├── requirements.txt
│   └── .env.example
├── ai/
├── data/
├── frontend/
├── notebooks/
├── tests/
├── docs/
├── models/
├── .gitignore
├── README.md
└── scripts/
```

## Phase 1 deliverables

- creates the project skeleton and directories
- defines a demo-friendly data-management layer
- loads sample image metadata from CSV
- validates images and metadata files
- preprocesses images by resizing and normalization
- prevents leakage by grouping records by cyclone ID during splits
- provides a local sample dataset generator for development only

## Important notes

- No real cyclone measurements or external APIs are used in this phase.
- The demo dataset is synthetic and clearly marked as such for development and testing purposes only.
- This is a baseline prototype and not an official warning system.
- Uploaded-image inference currently supports cyclone detection and the synthetic four-stage prototype classifier only. It does not produce an IMD eight-class result.
- Eight-class IMD probabilities on the dashboard are illustrative demo data; trained-model evaluation metrics are not exposed unless actual evaluation data is provided.

## Next step

The project is complete and ready for SIH presentation and demo showcase. See:
- [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for presentation walkthrough
- [SIH_SUBMISSION.md](SIH_SUBMISSION.md) for project summary
- [docs/deployment.md](docs/deployment.md) for local setup guide
