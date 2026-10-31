from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from ai.config.imd_classes import IMD_CLASSES
from utils.preprocessing import load_rgb_image
from utils.temporal_analysis import derive_evolution_status, parse_timestamp, sort_observations


ROOT = Path(__file__).resolve().parent
try:
    secret_api_url = st.secrets.get("CYCLO_SENSE_API_URL", "")
except StreamlitSecretNotFoundError:
    secret_api_url = ""
API_URL = (
    os.environ.get("CYCLO_SENSE_API_URL")
    or secret_api_url
    or "http://127.0.0.1:5000"
).rstrip("/")
METRICS_PATH = ROOT / "outputs" / "metrics.json"

st.set_page_config(
    page_title="CYCLO SENSE | Cyclone Intelligence",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: radial-gradient(ellipse at top left, #102c49 0%, #071321 48%, #06101b 100%); color: #edf6ff; }
    [data-testid="stSidebar"] { background: #091827; border-right: 1px solid #193b55; }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3 { color: #eff9ff; letter-spacing: -0.025em; }
    .eyebrow { color: #52dcf1; text-transform: uppercase; letter-spacing: .18em; font-size: .78rem; font-weight: 700; }
    .hero { padding: 1.6rem 1.8rem; margin: .25rem 0 1.1rem; border: 1px solid #1a4961; border-radius: 18px; background: linear-gradient(120deg, rgba(11,46,70,.94), rgba(8,26,43,.78)); box-shadow: 0 18px 60px rgba(0,0,0,.18); }
    .hero p { color: #9db7ca; margin-bottom: 0; }
    div[data-testid="stMetric"] { background: rgba(13,34,53,.88); border: 1px solid #20425a; padding: 1rem 1.1rem; border-radius: 14px; }
    div[data-testid="stMetricLabel"] { color: #9db7ca; }
    div[data-testid="stMetricValue"] { color: #53dcef; }
    .notice { border: 1px solid #22516a; background: #0b2336; padding: .9rem 1rem; border-radius: 12px; color: #c6deed; }
    .stButton > button { border-radius: 9px; border: 1px solid #28a9c6; background: linear-gradient(100deg,#0c7893,#106481); color: white; font-weight: 700; }
    .stButton > button:hover { border-color: #62eaff; color: white; }
    [data-testid="stFileUploader"] { border: 1px dashed #28758d; border-radius: 13px; background: rgba(9,31,48,.55); }
    </style>
    """,
    unsafe_allow_html=True,
)

if "observations" not in st.session_state:
    st.session_state.observations = []


def fetch_health() -> dict[str, Any]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        return {"status": "unavailable", "model_ready": False, "message": str(error)}


def _observation_time(item: dict[str, Any]) -> Any:
    return item.get("timestamp") or item.get("analysis_timestamp")


def get_observations() -> list[dict[str, Any]]:
    return sort_observations(list(st.session_state.observations))


def _format_confidence(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _metadata_from_csv(upload: Any, filenames: list[str]) -> dict[str, dict[str, Any]]:
    if upload is None:
        return {}
    try:
        frame = pd.read_csv(BytesIO(upload.getvalue()))
    except (pd.errors.ParserError, UnicodeError, ValueError) as error:
        st.warning(f"Metadata CSV could not be read; predictions can continue without it: {error}")
        return {}
    if "filename" not in frame.columns:
        st.warning("Metadata CSV needs a 'filename' column to match rows to uploaded images.")
        return {}

    known = set(filenames)
    metadata: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        filename = str(row.get("filename", "")).strip()
        if filename not in known:
            continue
        record: dict[str, Any] = {}
        for field in ("timestamp", "cyclone_id", "group_id", "latitude", "longitude"):
            value = row.get(field)
            if value is not None and not pd.isna(value) and str(value).strip():
                record[field] = value
        metadata[filename] = record

    unmatched = known - set(metadata)
    if unmatched:
        st.warning(
            "No metadata row matched: " + ", ".join(sorted(unmatched))
            + ". Timestamps may be read from filenames; coordinates remain unavailable."
        )
    return metadata


def _add_observation_details(
    result: dict[str, Any],
    image_bytes: bytes,
) -> dict[str, Any]:
    item = dict(result)
    item["_image_bytes"] = image_bytes
    return item


def _render_probabilities(
    probabilities: dict[str, float],
    title: str = "Prototype stage probabilities",
) -> None:
    figure = go.Figure(
        go.Bar(
            x=[float(probabilities.get(name, 0.0)) * 100 for name in IMD_CLASSES],
            y=list(IMD_CLASSES),
            orientation="h",
            marker=dict(
                color=[float(probabilities.get(name, 0.0)) * 100 for name in IMD_CLASSES],
                colorscale=[[0, "#14506b"], [1, "#45d8ec"]],
                line=dict(width=0),
            ),
            text=[f"{float(probabilities.get(name, 0.0)) * 100:.1f}%" for name in IMD_CLASSES],
            textposition="outside",
            cliponaxis=False,
        )
    )
    figure.update_layout(
        height=380,
        margin=dict(l=8, r=35, t=14, b=8),
        xaxis=dict(title="Model probability (%)", range=[0, 108], gridcolor="#1b3448"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dceaf5"),
        showlegend=False,
    )
    st.markdown(f"**{title}**")
    st.plotly_chart(figure, use_container_width=True)


def _render_prediction(item: dict[str, Any]) -> None:
    st.markdown(f"#### {item.get('filename', 'Uploaded image')}")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        image_bytes = item.get("_image_bytes")
        if image_bytes is None:
            st.info("Image preview is unavailable.")
        else:
            try:
                st.image(load_rgb_image(image_bytes), use_container_width=True)
            except ValueError:
                st.info("Image preview is unavailable.")
    with right:
        if item.get("status") == "error":
            st.error(item.get("error", "Image analysis failed."))
            return
        st.caption(item.get("data_mode", "Prototype model output"))
        st.warning(item.get("model_provenance", "This model output is not validated on real satellite imagery."))
        if item.get("cyclone_detected") is False:
            st.error("Non-Cyclone Image")
            st.metric("Detection confidence", _format_confidence(item.get("detection_confidence")))
            st.caption("Prototype stage classification was skipped because this model did not detect a cyclone.")
            return
        stage_probabilities = item.get("stage_probabilities") or item.get("probabilities")
        if not stage_probabilities:
            st.warning("The API did not return prototype stage probabilities.")
            return

        st.success("Cyclone detected by the prototype model")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prototype stage", item.get("classification") or item.get("predicted_class") or "—")
        c2.metric("Confidence", _format_confidence(item.get("confidence")))
        c3.metric("Detection", _format_confidence(item.get("detection_confidence")))
        st.markdown("**Top prototype stages**")
        for rank, prediction in enumerate(item.get("top_3", []), start=1):
            st.write(f"{rank}. {prediction['class_name']} — {_format_confidence(prediction['probability'])}")
        _render_probabilities(stage_probabilities)
        if item.get("explanation_image"):
            st.image(item["explanation_image"], caption="Grad-CAM for the prototype stage prediction")


def _render_temporal(observations: list[dict[str, Any]]) -> None:
    st.markdown('<p class="eyebrow">Observation sequence</p>', unsafe_allow_html=True)
    st.title("Temporal Analysis")
    cyclone_observations = [
        item for item in observations
        if item.get("cyclone_detected") is True and item.get("predicted_class") in IMD_CLASSES
    ]
    if not cyclone_observations:
        st.info("No cyclone predictions are recorded in this browser session yet.")
        return

    trend = derive_evolution_status(cyclone_observations)
    st.metric("Derived intensity trend", trend)
    ordered = sort_observations(cyclone_observations)
    chart_rows = [
        item for item in ordered
        if parse_timestamp(_observation_time(item)) is not None
    ]
    if len(chart_rows) >= 2:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=[parse_timestamp(_observation_time(item)) for item in chart_rows],
                y=[IMD_CLASSES.index(item["predicted_class"]) for item in chart_rows],
                mode="lines+markers",
                line=dict(color="#42d9ee", width=3),
                marker=dict(size=11, color="#42d9ee", line=dict(color="#d9fbff", width=1)),
                text=[item.get("predicted_class") for item in chart_rows],
                customdata=[[_format_confidence(item.get("confidence"))] for item in chart_rows],
                hovertemplate="%{x}<br>%{text}<br>Confidence: %{customdata[0]}<extra></extra>",
            )
        )
        figure.update_layout(
            height=390,
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(IMD_CLASSES))),
                ticktext=list(IMD_CLASSES),
                autorange="reversed",
                title="Predicted IMD class",
                gridcolor="#1b3448",
            ),
            xaxis=dict(title="Observation time", gridcolor="#1b3448"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dceaf5"),
            margin=dict(l=10, r=10, t=14, b=10),
        )
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("Add verified timestamps in the metadata CSV or filenames to plot intensity against time.")

    timeline = []
    for item in ordered:
        timeline.append(
            {
                "Timestamp": _observation_time(item) or "Timestamp unavailable",
                "Image": item.get("filename"),
                "Predicted intensity": item.get("predicted_class"),
                "Confidence": _format_confidence(item.get("confidence")),
                "Cyclone detected": item.get("cyclone_detected"),
            }
        )
    st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)


def _render_track(observations: list[dict[str, Any]]) -> None:
    st.markdown('<p class="eyebrow">Geospatial observations</p>', unsafe_allow_html=True)
    st.title("Cyclone Track")
    track = []
    for item in sort_observations(observations):
        try:
            latitude = float(item["latitude"])
            longitude = float(item["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            track.append({**item, "latitude": latitude, "longitude": longitude})
    if not track:
        st.info("No valid latitude/longitude metadata is available. Add coordinates to the metadata CSV.")
        return

    figure = go.Figure()
    if len(track) > 1:
        figure.add_trace(
            go.Scattergeo(
                lat=[item["latitude"] for item in track],
                lon=[item["longitude"] for item in track],
                mode="lines",
                line=dict(width=2, color="#55dced"),
                name="Observed sequence",
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scattergeo(
            lat=[item["latitude"] for item in track],
            lon=[item["longitude"] for item in track],
            mode="markers+text",
            text=[str(index + 1) for index in range(len(track))],
            textposition="top center",
            marker=dict(size=12, color="#4de0f1", line=dict(width=1, color="#ecfcff")),
            customdata=[
                [
                    item.get("timestamp") or "Timestamp unavailable",
                    item.get("predicted_class") or "Non-cyclone",
                    _format_confidence(item.get("confidence")),
                    item.get("filename", ""),
                ]
                for item in track
            ],
            hovertemplate=(
                "%{customdata[3]}<br>%{customdata[0]}<br>"
                "%{lat:.3f}, %{lon:.3f}<br>%{customdata[1]} "
                "(%{customdata[2]})<extra></extra>"
            ),
            name="Observed locations",
        )
    )
    figure.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#10283b",
        showocean=True,
        oceancolor="#071827",
        showcountries=True,
        countrycolor="#345266",
        showcoastlines=True,
        coastlinecolor="#4a728a",
        bgcolor="rgba(0,0,0,0)",
        fitbounds="locations",
    )
    figure.update_layout(
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dceaf5"),
        margin=dict(l=0, r=0, t=12, b=0),
        legend=dict(font=dict(color="#dceaf5")),
    )
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Timestamp": item.get("timestamp") or "Timestamp unavailable",
                    "Latitude": item["latitude"],
                    "Longitude": item["longitude"],
                    "Intensity": item.get("predicted_class") or "Non-cyclone",
                    "Confidence": _format_confidence(item.get("confidence")),
                }
                for item in track
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_dashboard(health: dict[str, Any], observations: list[dict[str, Any]]) -> None:
    st.markdown('<p class="eyebrow">SIH26070 · Cyclone Intelligence</p>', unsafe_allow_html=True)
    st.markdown(
        "<div class='hero'><h1>CYCLO SENSE</h1>"
        "<p>Satellite image analysis, cyclone detection, and eight-class IMD intensity insights.</p></div>",
        unsafe_allow_html=True,
    )
    if not health.get("model_ready"):
        if health.get("status") == "unavailable":
            st.error(f"Flask backend unavailable: {health.get('message')}")
        else:
            st.warning(health.get("message", "Model not trained yet. Run train_model.py."))
    else:
        st.success("ResNet-18 prototype checkpoints are available. Results are not validated on real satellite imagery.")

    ordered = get_observations()
    latest = ordered[-1] if ordered else {}
    latest_time = _observation_time(latest) or "—"
    latest_class = latest.get("predicted_class") or (
        "Non-Cyclone Image" if latest.get("cyclone_detected") is False else "—"
    )
    st.markdown("### Operational snapshot")
    cards = st.columns(5)
    cards[0].metric("Current intensity", latest_class)
    cards[1].metric("Confidence", _format_confidence(latest.get("confidence")))
    cards[2].metric("Evolution", derive_evolution_status(ordered))
    cards[3].metric("Observations", len(ordered))
    cards[4].metric("Latest timestamp", str(latest_time)[:19])
    if latest.get("_image_bytes"):
        st.markdown("### Recent satellite image")
        st.image(latest["_image_bytes"], caption=latest.get("filename", "Latest analyzed image"), width=520)
    if ordered:
        st.markdown("### Recent model observations")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Timestamp": _observation_time(item) or "Unavailable",
                        "Image": item.get("filename"),
                        "Cyclone detected": item.get("cyclone_detected"),
                        "Intensity": item.get("predicted_class") or "Not classified",
                        "Confidence": _format_confidence(item.get("confidence")),
                    }
                    for item in reversed(ordered[-8:])
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No images have been analyzed in this session yet. No predictions or metrics are fabricated.")


def _render_analysis() -> None:
    st.markdown('<p class="eyebrow">Real model inference</p>', unsafe_allow_html=True)
    st.title("Analyze Satellite Images")
    st.write("Upload JPG, JPEG, or PNG images. Non-cyclone images are rejected before intensity classification.")
    images = st.file_uploader(
        "Select one or more satellite images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="satellite_uploads",
    )
    metadata_file = st.file_uploader(
        "Optional observation metadata CSV",
        type=["csv"],
        accept_multiple_files=False,
        help="Columns: filename, timestamp, cyclone_id, latitude, longitude.",
        key="observation_metadata",
    )

    if images:
        previews = st.columns(min(len(images), 4))
        for index, image in enumerate(images):
            with previews[index % len(previews)]:
                st.caption(image.name)
                try:
                    st.image(load_rgb_image(image.getvalue()), use_container_width=True)
                except ValueError as error:
                    st.error(str(error))

    if st.button("Run cyclone analysis", type="primary", disabled=not images):
        names = [image.name for image in images]
        metadata = _metadata_from_csv(metadata_file, names)
        files = []
        image_bytes_by_name: dict[str, bytes] = {}
        local_errors: list[dict[str, Any]] = []
        for image in images:
            image_bytes = image.getvalue()
            image_bytes_by_name[image.name] = image_bytes
            try:
                load_rgb_image(image_bytes)
            except ValueError as error:
                local_errors.append(
                    {"filename": image.name, "status": "error", "error": str(error)}
                )
                continue
            mime = "image/png" if image.name.casefold().endswith(".png") else "image/jpeg"
            files.append(("files", (image.name, image_bytes, mime)))

        results: list[dict[str, Any]] = list(local_errors)
        if files:
            form_metadata = [
                {"filename": image.name, **metadata.get(image.name, {})}
                for image in images
                if any(upload[1][0] == image.name for upload in files)
            ]
            with st.spinner("Running the ResNet-18 prototype inference API…"):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/multiple",
                        files=files,
                        data={"metadata": json.dumps(form_metadata)},
                        timeout=240,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    for item in payload.get("observations", []):
                        name = item.get("filename", "")
                        if name in image_bytes_by_name:
                            item = _add_observation_details(item, image_bytes_by_name[name])
                        results.append(item)
                except requests.Timeout:
                    st.error("The inference API timed out. Check that Flask is running and the model is loaded.")
                except requests.ConnectionError:
                    st.error(f"Cannot connect to Flask at {API_URL}. Start the backend in another terminal.")
                except requests.HTTPError as error:
                    detail = error.response.text if error.response is not None else str(error)
                    st.error(f"Inference API error: {detail}")
                except (requests.RequestException, ValueError) as error:
                    st.error(f"Could not read the inference API response: {error}")

        if results:
            st.session_state.observations.extend(results)
            st.session_state.last_results = results

    prior_results = st.session_state.get("last_results", [])
    if prior_results:
        st.markdown("---")
        st.markdown("### Latest analysis results")
        for item in prior_results:
            _render_prediction(item)
            st.markdown("---")


def _render_analytics() -> None:
    st.markdown('<p class="eyebrow">Held-out evaluation</p>', unsafe_allow_html=True)
    st.title("Model Analytics")
    if not METRICS_PATH.is_file():
        st.info("No actual evaluation metrics are available yet. Train MobileNetV2 on verified data first.")
        return
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read saved metrics: {error}")
        return
    a, b, c, d = st.columns(4)
    a.metric("8-class accuracy", f"{metrics['intensity_accuracy'] * 100:.2f}%")
    b.metric("Macro precision", f"{metrics['intensity_macro_precision'] * 100:.2f}%")
    c.metric("Macro recall", f"{metrics['intensity_macro_recall'] * 100:.2f}%")
    d.metric("Macro F1", f"{metrics['intensity_macro_f1'] * 100:.2f}%")
    st.caption(
        f"Evaluation uses {metrics['cyclone_test_rows']} cyclone images from the held-out "
        f"{metrics['test_rows']}-image test split. Metrics are read from {METRICS_PATH}."
    )
    st.markdown("### Per-class test report")
    report = metrics.get("intensity_classification_report", {})
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Class": name,
                    "Precision": values.get("precision"),
                    "Recall": values.get("recall"),
                    "F1-score": values.get("f1-score"),
                    "Support": values.get("support"),
                }
                for name, values in report.items()
                if name in IMD_CLASSES
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    matrix_path = ROOT / "outputs" / "charts" / "confusion_matrix.png"
    if matrix_path.is_file():
        st.image(str(matrix_path), caption="Saved held-out 8-class confusion matrix")


def main() -> None:
    health = fetch_health()
    observations = get_observations()
    st.sidebar.markdown(
        "<div class='eyebrow'>SIH26070</div><h2 style='margin-top:.2rem'>CYCLO SENSE</h2>",
        unsafe_allow_html=True,
    )
    pages = (
        "Dashboard",
        "Analyze Satellite Image",
        "Temporal Analysis",
        "Cyclone Track",
        "Analytics",
        "About",
    )
    page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")
    st.sidebar.markdown("---")
    if health.get("model_ready"):
        st.sidebar.success("PROTOTYPE MODELS READY")
    elif health.get("status") == "unavailable":
        st.sidebar.error("BACKEND OFFLINE")
    else:
        st.sidebar.warning("MODEL NOT TRAINED")
    st.sidebar.caption("No demo probabilities are mixed with real inference.")

    if page == "Dashboard":
        _render_dashboard(health, observations)
    elif page == "Analyze Satellite Image":
        _render_analysis()
    elif page == "Temporal Analysis":
        _render_temporal(observations)
    elif page == "Cyclone Track":
        _render_track(observations)
    elif page == "Analytics":
        _render_analytics()
    else:
        st.markdown('<p class="eyebrow">System overview</p>', unsafe_allow_html=True)
        st.title("About CYCLO SENSE")
        st.markdown(
            """
            **CYCLO SENSE** is a Smart India Hackathon SIH26070 prototype for satellite-image cyclone analysis.

            - The available ResNet-18 detector and four-stage classifier were trained on generated synthetic images.
            - Their outputs and Grad-CAM maps are prototype-only and have not been validated on real satellite imagery.
            - No trained eight-class IMD, intensity-forecasting, track-forecasting, or risk model is available.
            - Observations in this dashboard are generated only from successful API inference in the current browser session.
            - Time-series trends require timestamps. Track plots require valid latitude and longitude metadata.

            This is a research prototype, not an official meteorological warning service.
            """
        )


if __name__ == "__main__":
    main()
