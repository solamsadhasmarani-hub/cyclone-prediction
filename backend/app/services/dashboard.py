from __future__ import annotations

from copy import deepcopy


IMD_INTENSITY_CLASSES = [
    "Low Pressure Area",
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Storm",
    "Super Cyclonic Storm",
]

DEMO_DASHBOARD = {
    "health": {"status": "ok", "app": "CYCLO SENSE", "environment": "demo"},
    "model_status": {
        "detection_model_ready": True,
        "classification_model_ready": True,
        "intensity_model_ready": True,
        "track_model_ready": True,
        "demo_mode": True,
    },
    "model_information": {
        "proposed_architecture": "MobileNetV2 + transfer learning + custom classification head",
        "input_size": "224 × 224",
        "output_classes": IMD_INTENSITY_CLASSES,
        "proposed_model_implemented": False,
        "current_prototype_architecture": "ResNet-18",
        "current_prototype_output": "4 synthetic cyclone development stages",
    },
    "cyclones": [
        {
            "cyclone_id": "DEMO_01",
            "cyclone_name": "Demo Cyclone 01",
            "status": "demo",
            "category": "Severe Cyclonic Storm",
            "center": "14.2N, 82.3E",
        },
        {
            "cyclone_id": "DEMO_02",
            "cyclone_name": "Demo Cyclone 02",
            "status": "demo",
            "category": "Cyclonic Storm",
            "center": "13.8N, 83.2E",
        },
    ],
    "selected_cyclone": {
        "cyclone_id": "DEMO_01",
        "cyclone_name": "Demo Cyclone 01",
        "detection": True,
        "detection_confidence": 0.94,
        "imd_classification": "Severe Cyclonic Storm",
        "classification_confidence": 0.94,
        "wind_speed_kmh": 98,
        "timestamp": "2026-10-01T16:00:00Z",
        "latitude": 14.2,
        "longitude": 82.3,
    },
    "imd_intensity_probabilities": [
        {"class_name": class_name, "probability": probability}
        for class_name, probability in zip(
            IMD_INTENSITY_CLASSES,
            [2, 3, 5, 8, 76, 4, 1, 1],
        )
    ],
    "probabilities_source": "demo",
    "timestamped_observations": [
        {
            "timestamp": "2026-10-01T13:00:00Z",
            "time_label": "13:00 UTC",
            "classification": "Cyclonic Storm",
            "latitude": 12.4,
            "longitude": 80.9,
        },
        {
            "timestamp": "2026-10-01T14:00:00Z",
            "time_label": "14:00 UTC",
            "classification": "Cyclonic Storm",
            "latitude": 13.1,
            "longitude": 81.5,
        },
        {
            "timestamp": "2026-10-01T16:00:00Z",
            "time_label": "16:00 UTC",
            "classification": "Severe Cyclonic Storm",
            "latitude": 14.2,
            "longitude": 82.3,
        },
    ],
    "evolution_source": "derived from demo timestamped observations",
    "observed_track": [
        {
            "timestamp": "2026-10-01T13:00:00Z",
            "time_label": "13:00 UTC",
            "latitude": 12.4,
            "longitude": 80.9,
        },
        {
            "timestamp": "2026-10-01T14:00:00Z",
            "time_label": "14:00 UTC",
            "latitude": 13.1,
            "longitude": 81.5,
        },
        {
            "timestamp": "2026-10-01T16:00:00Z",
            "time_label": "16:00 UTC",
            "latitude": 14.2,
            "longitude": 82.3,
        },
    ],
    "predicted_track": [
        {"hours": 6, "time_label": "+6h", "latitude": 15.0, "longitude": 83.1},
        {"hours": 12, "time_label": "+12h", "latitude": 15.8, "longitude": 84.0},
        {"hours": 18, "time_label": "+18h", "latitude": 16.6, "longitude": 84.8},
    ],
    "intensity_forecast": [
        {"name": "+0h", "value": 98},
        {"name": "+6h", "value": 101},
        {"name": "+12h", "value": 104},
        {"name": "+18h", "value": 108},
        {"name": "+24h", "value": 110},
    ],
    "track_prediction": [
        {"hours": 0, "latitude": 14.2, "longitude": 82.3},
        {"hours": 6, "latitude": 15.0, "longitude": 83.1},
        {"hours": 12, "latitude": 15.8, "longitude": 84.0},
        {"hours": 18, "latitude": 16.6, "longitude": 84.8},
    ],
    "predictions": [
        {"type": "Cyclone Detection", "result": "YES"},
        {"type": "IMD Classification", "result": "Severe Cyclonic Storm"},
        {"type": "Confidence", "result": "94%"},
        {"type": "Wind Speed", "result": "98 km/h"},
        {"type": "Evolution", "result": "Strengthening"},
    ],
    "historical_cyclones": [
        {
            "name": "Historical Demo Cyclone A",
            "classification": "Cyclonic Storm",
            "recorded_at": "2024-10-12T09:00:00Z",
            "data_type": "historical demo record",
        },
        {
            "name": "Historical Demo Cyclone B",
            "classification": "Very Severe Cyclonic Storm",
            "recorded_at": "2023-11-04T12:00:00Z",
            "data_type": "historical demo record",
        },
    ],
    "summary": {
        "active_cyclone_count": 2,
        "latest_analysis_time": "16:00 UTC",
        "detection_status": "YES",
        "model_status_label": "Demo ready",
    },
    "model_metrics": None,
    "alerts": [
        {
            "title": "Severe Cyclonic Storm",
            "details": "Strengthening · 94% demo confidence",
            "data_type": "DEMO ALERT",
        }
    ],
}


def derive_evolution_status(observations: list[dict[str, object]]) -> str:
    """Derive trend from the latest two timestamp-ordered IMD class observations."""
    class_rank = {name: rank for rank, name in enumerate(IMD_INTENSITY_CLASSES)}
    ordered = sorted(observations, key=lambda observation: str(observation["timestamp"]))
    ranks = [
        class_rank.get(str(observation["classification"]))
        for observation in ordered
    ]
    known_ranks = [rank for rank in ranks if rank is not None]
    if len(known_ranks) < 2:
        return "INSUFFICIENT TEMPORAL DATA"
    if known_ranks[-1] > known_ranks[-2]:
        return "STRENGTHENING"
    if known_ranks[-1] < known_ranks[-2]:
        return "WEAKENING"
    return "PEAK"


def build_demo_dashboard() -> dict[str, object]:
    """Return an isolated copy of the centralized, explicitly synthetic dashboard payload."""
    dashboard = deepcopy(DEMO_DASHBOARD)
    evolution = derive_evolution_status(dashboard["timestamped_observations"])
    dashboard["evolution_status"] = evolution
    predictions = dashboard["predictions"]
    evolution_prediction = next((item for item in predictions if item["type"] == "Evolution"), None)
    if evolution_prediction is not None:
        evolution_prediction["result"] = evolution.title()
    alerts = dashboard["alerts"]
    if alerts:
        alerts[0]["details"] = f"{evolution.title()} · 94% demo confidence"
    return dashboard
