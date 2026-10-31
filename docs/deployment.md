# SIH26070 deployment and demo guide

This project is a demo-safe cyclone AI prototype. It uses synthetic sample data and does not provide operational cyclone warnings.

## Publish both dashboards

The React/Vite and Streamlit dashboards are separate apps. Deploy them with the same GitHub repository connected to Vercel, Render, and Streamlit Community Cloud.

### 1. Deploy both APIs on Render

1. Push the project to GitHub.
2. In Render, choose **New → Blueprint**, connect the repository, and deploy its `render.yaml`. This creates the FastAPI and Flask API services.
3. Wait for both services to finish deploying. Check the FastAPI service at `/api/healthz` and the Flask service at `/health`.

The Flask service deliberately uses `deployment/flask-api-requirements.txt`, which omits TensorFlow because no trained model is included. The API reports that inference is unavailable until a trained model and its metadata are deployed. If you later add a trained MobileNetV2 model, update that requirements file to include the supported TensorFlow dependency before deploying.

### 2. Deploy the React/Vite dashboard on Vercel

1. Import the same GitHub repository into Vercel.
2. Set **Root Directory** to `frontend` and use the Vite build defaults (`npm run build`, output directory `dist`).
3. Add the environment variable `VITE_API_BASE_URL` with the deployed FastAPI service URL, for example `https://cyclo-sense-fastapi.onrender.com`.
4. Deploy and copy the resulting Vercel domain.
5. In the Render FastAPI service, set `CORS_ORIGINS` to that exact Vercel origin (scheme and hostname only, no trailing slash), for example `https://your-dashboard.vercel.app`. Keep multiple origins comma-separated if needed, then redeploy the service.

### 3. Deploy the Streamlit dashboard on Streamlit Community Cloud

1. Create a new app from the same GitHub repository and select `streamlit_app/app.py` as the app file. Its adjacent `requirements.txt` installs only the dashboard dependencies.
2. In the app's **Settings → Secrets**, add:

   ```toml
   CYCLO_SENSE_API_URL = "https://cyclo-sense-flask-api.onrender.com"
   ```

3. Deploy the app and use the URL Streamlit Community Cloud provides.

The free hosting tiers may sleep when idle, so the first request after a pause can take a minute. Both dashboards clearly identify demo data; uploaded-image inference remains unavailable until the respective trained model files are provided.

## Local startup

For the React/Vite + FastAPI demo stack, run the project launcher from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\start_demo.py
```

This launches the FastAPI API at `http://localhost:8000` and the React dashboard at `http://localhost:5173`.

For the separate Streamlit + Flask demo stack, start the Flask API in one terminal and the dashboard in another:

```powershell
.\.venv\Scripts\python.exe -m backend.api
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Validation

Run the API smoke validation with:

```powershell
.\.venv\Scripts\python.exe scripts\phase11_validation.py
```

Run the Python tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Build the frontend with:

```powershell
npm --prefix frontend run build
```
