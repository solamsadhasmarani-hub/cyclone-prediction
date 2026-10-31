# SIH26070 deployment and demo guide

This project is a demo-safe cyclone AI prototype. It uses synthetic sample data and does not provide operational cyclone warnings.

## Free-first deployment: Streamlit dashboard + Flask inference API

This project has two separate dashboard stacks. For the existing Streamlit URL and image-upload feature, deploy the Streamlit dashboard on Streamlit Community Cloud and its Flask inference API on Render. The React/Vite + FastAPI stack is separate and is not required for this deployment. The Render blueprint intentionally creates only the Flask API service.

Both platforms offer no-cost options, but their free services can sleep when idle. Render documents ephemeral service storage and notes that its free instances are intended for hobby/testing, not production. Do not store uploads or other data permanently on the API filesystem. No paid plan is selected by this configuration.

### 1. Deploy the Flask API on Render

1. Push this repository to GitHub, including `render.yaml`, `deployment/flask-api-requirements.txt`, and both tracked files in `models/`.
2. In Render, choose **New → Blueprint**, connect the repository, review the `cyclo-sense-flask-api` service, and select the **Free** plan. Do not upgrade unless you explicitly choose a paid plan.
3. Wait for the service deployment to complete. Its public API URL will look like `https://cyclo-sense-flask-api.onrender.com`. Open `<your-api-url>/health`; verify `model_ready` and the detector/classifier readiness flags are true.

The Render build installs Flask/API dependencies, then the official CPU-only PyTorch and torchvision wheels. It uses Python 3.11.11, Gunicorn, Render's `$PORT`, and `/health` for service checks. Build and inference are CPU-only. Free-service sleeping can make the first inference request slow; model startup or inference can exceed free resources, so verify with a real upload before sharing the dashboard link.

The two ResNet-18 checkpoints were trained on generated synthetic demo images and are not validated on real satellite imagery. They provide prototype-only detection and four-stage outputs. No trained eight-class IMD, intensity-forecasting, track-forecasting, or risk model is included.

### 2. Deploy the Streamlit dashboard

1. Sign in to Streamlit Community Cloud with GitHub and choose **Create app**.
2. Select this repository, its `main` branch, and `streamlit_app/app.py` as the app file.
3. Before or after the first deployment, open the app's **Settings → Secrets** and add the URL Render gave your Flask service:

   ```toml
   CYCLO_SENSE_API_URL = "https://your-render-api-url"
   ```

4. Save the secret, deploy/reboot the app, and open the Streamlit HTTPS URL it provides. The dashboard-only requirements are in `streamlit_app/requirements.txt`; the wrapper runs the existing root `app.py`.

The dashboard calls the API server-to-server over HTTPS, so browser CORS is not needed for this Streamlit deployment. `VITE_API_BASE_URL` is only used by the separate React dashboard.

### 3. Verify upload and auto-deploy

1. In Streamlit, confirm the sidebar reports that the prototype checkpoints are ready.
2. Upload a JPG or PNG in **Analyze Satellite Image** and confirm the API returns a result and Grad-CAM when the prototype detector accepts the image.
3. Confirm the result shows the synthetic-training warning. Do not interpret the four prototype stages as IMD intensity or operational warnings.
4. After changes are pushed to the GitHub `main` branch, Streamlit Community Cloud and the Render Blueprint can automatically redeploy from that branch. Check both service logs if a deploy fails.

These free services may sleep when idle. Render loses local filesystem changes on restart/redeploy, so do not treat uploaded files or local API storage as persistent.

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
