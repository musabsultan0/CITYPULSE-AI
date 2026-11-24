from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.app.core.config import ROOT_DIR, settings

TRAFFIC_FEATURES = [
    "hour",
    "day_of_week",
    "vehicle_count",
    "average_speed",
    "road_capacity",
    "rainfall_mm",
    "special_event",
]

FLOOD_FEATURES = [
    "rainfall_mm_hour",
    "water_level_m",
    "drainage_capacity_pct",
    "previous_24h_rainfall_mm",
    "soil_moisture_pct",
    "elevation_m",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _resolve_model_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return ROOT_DIR / p


@lru_cache(maxsize=4)
def _load_joblib_model(path_str: str) -> Any:
    path = _resolve_model_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    model = joblib.load(path)
    # Single-row inference inside FastAPI worker threads must not spawn joblib worker pools
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


def get_traffic_model() -> Any:
    return _load_joblib_model(settings.traffic_model_path)


def get_flood_model() -> Any:
    return _load_joblib_model(settings.flood_model_path)


@lru_cache(maxsize=2)
def get_flood_metrics() -> dict[str, Any]:
    path = _resolve_model_path(settings.flood_metrics_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def predict_traffic_congestion(
    features: dict[str, Any],
    data_source: str = "user_supplied",
    zone: str = "Zone A",
) -> dict[str, Any]:
    """Run inference using the trained Traffic Random Forest Regressor."""
    clean_inputs = {
        "hour": int(features["hour"]),
        "day_of_week": int(features["day_of_week"]),
        "vehicle_count": float(features["vehicle_count"]),
        "average_speed": float(features["average_speed"]),
        "road_capacity": float(features["road_capacity"]),
        "rainfall_mm": float(features["rainfall_mm"]),
        "special_event": int(features["special_event"]),
    }
    model_path = _resolve_model_path(settings.traffic_model_path)
    try:
        model = get_traffic_model()
        df = pd.DataFrame([clean_inputs], columns=TRAFFIC_FEATURES)
        raw_pred = float(model.predict(df)[0])
        predicted_pct = round(_clamp(raw_pred, 0.0, 100.0), 2)
        importances = {}
        if hasattr(model, "feature_importances_"):
            importances = {
                feat: round(float(val), 4)
                for feat, val in sorted(
                    zip(TRAFFIC_FEATURES, model.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }
        model_available = True
        method = "random_forest_regressor"
    except Exception as exc:
        # Graceful fallback if model file is missing
        load_ratio = clean_inputs["vehicle_count"] / max(clean_inputs["road_capacity"], 1.0)
        fallback = 35.0 * _clamp(load_ratio, 0.0, 2.0) + 20.0 * (1.0 - _clamp(clean_inputs["average_speed"] / 80.0)) + 10.0 * _clamp(clean_inputs["rainfall_mm"] / 30.0)
        predicted_pct = round(_clamp(fallback, 0.0, 100.0), 2)
        importances = {}
        model_available = False
        method = f"fallback_heuristic ({exc})"

    if predicted_pct >= 70.0:
        status = "HIGH"
    elif predicted_pct >= 40.0:
        status = "MODERATE"
    else:
        status = "LOW"

    try:
        rel_artifact = str(model_path.relative_to(ROOT_DIR).as_posix())
    except ValueError:
        rel_artifact = str(model_path)

    return {
        "zone": zone,
        "predicted_congestion_pct": predicted_pct,
        "traffic_status": status,
        "inputs": clean_inputs,
        "model_info": {
            "model_name": "Traffic Congestion Random Forest Regressor",
            "model_available": model_available,
            "method": method,
            "artifact_path": rel_artifact,
            "feature_importances": importances,
        },
        "data_provenance": {
            "input_source": data_source,
            "is_synthetic_training_data": True,
            "training_dataset": "ml/traffic/datasets/traffic_data.csv (synthetic)",
        },
        "disclaimer": "Model trained on synthetic traffic data for prototype decision support; not validated for real-world traffic control.",
    }


def derive_traffic_features_from_readings(
    readings: dict[str, Any],
    scenario: str = "normal",
) -> tuple[dict[str, Any], str]:
    """Derive 7 Traffic ML features from general sensor readings when explicit features are not supplied."""
    now = datetime.now()
    density = float(readings.get("traffic_density_pct", 50.0))
    rainfall = float(readings.get("rainfall_mm_hour", 0.0))
    source = str(readings.get("source", "simulated_demo"))

    hour = 18 if scenario == "traffic_peak" else (8 if scenario == "heavy_rain" else now.hour)
    road_capacity = 600.0
    vehicle_count = round(_clamp(density / 100.0, 0.05, 1.0) * road_capacity * 1.65, 1)
    average_speed = round(max(10.0, 75.0 - 0.62 * density - 0.12 * min(rainfall, 50.0)), 1)
    special_event = 1 if scenario == "traffic_peak" else 0

    derived = {
        "hour": hour,
        "day_of_week": now.weekday(),
        "vehicle_count": vehicle_count,
        "average_speed": average_speed,
        "road_capacity": road_capacity,
        "rainfall_mm": min(rainfall, 30.0),
        "special_event": special_event,
    }
    return derived, f"derived_from_{source}"


def predict_flood_risk(
    features: dict[str, Any],
    data_source: str = "user_supplied",
    zone: str = "Zone A",
) -> dict[str, Any]:
    """Run inference using the trained Flood Risk Classifier."""
    clean_inputs = {
        "rainfall_mm_hour": float(features["rainfall_mm_hour"]),
        "water_level_m": float(features["water_level_m"]),
        "drainage_capacity_pct": float(features["drainage_capacity_pct"]),
        "previous_24h_rainfall_mm": float(features["previous_24h_rainfall_mm"]),
        "soil_moisture_pct": float(features["soil_moisture_pct"]),
        "elevation_m": float(features["elevation_m"]),
    }
    model_path = _resolve_model_path(settings.flood_model_path)
    metrics_meta = get_flood_metrics()
    try:
        model = get_flood_model()
        df = pd.DataFrame([clean_inputs], columns=FLOOD_FEATURES)
        prob = float(model.predict_proba(df)[0, 1])
        prob = _clamp(prob, 0.0, 1.0)
        model_available = True
        method = metrics_meta.get("selected_model", type(model).__name__)
    except Exception as exc:
        latent = (
            0.40 * _clamp(clean_inputs["rainfall_mm_hour"] / 140.0)
            + 0.30 * _clamp(clean_inputs["water_level_m"] / 2.2)
            + 0.20 * _clamp((100.0 - clean_inputs["drainage_capacity_pct"]) / 100.0)
            + 0.10 * _clamp(clean_inputs["previous_24h_rainfall_mm"] / 220.0)
        )
        prob = _clamp(latent, 0.0, 1.0)
        model_available = False
        method = f"fallback_heuristic ({exc})"

    risk_pct = round(prob * 100.0, 2)
    flood_predicted = prob >= 0.5
    if risk_pct >= 70.0:
        risk_level = "HIGH"
    elif risk_pct >= 40.0:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    try:
        rel_artifact = str(model_path.relative_to(ROOT_DIR).as_posix())
    except ValueError:
        rel_artifact = str(model_path)

    return {
        "zone": zone,
        "flood_probability": round(prob, 4),
        "flood_risk_pct": risk_pct,
        "flood_predicted": flood_predicted,
        "risk_level": risk_level,
        "inputs": clean_inputs,
        "model_info": {
            "model_name": "Flood Risk ML Classifier",
            "selected_algorithm": method,
            "model_available": model_available,
            "artifact_path": rel_artifact,
            "evaluation_metrics": metrics_meta.get("selected_metrics", {}),
            "feature_importances": metrics_meta.get("feature_importances", {}),
        },
        "data_provenance": {
            "input_source": data_source,
            "is_synthetic_training_data": True,
            "training_dataset": "ml/flood/datasets/flood_data.csv (synthetic)",
        },
        "disclaimer": "Flood ML model is trained on synthetic data for prototype demonstration only; never use for real flood warnings or evacuations.",
    }


def derive_flood_features_from_readings(
    readings: dict[str, Any],
    zone: str = "Zone A",
) -> tuple[dict[str, Any], str]:
    """Derive 6 Flood ML features from standard sensor readings."""
    rain = float(readings.get("rainfall_mm_hour", 0.0))
    water = float(readings.get("water_level_m", 0.1))
    drainage = float(readings.get("drainage_capacity_pct", 60.0))
    source = str(readings.get("source", "simulated_demo"))

    zone_elevation_map = {"Zone A": 12.0, "Zone B": 28.0, "Zone C": 45.0}
    elevation = zone_elevation_map.get(zone, 20.0)
    prev_24h = round(min(300.0, rain * 1.45 + 15.0), 1)
    soil_moisture = round(min(100.0, 32.0 + 0.35 * prev_24h), 1)

    derived = {
        "rainfall_mm_hour": rain,
        "water_level_m": water,
        "drainage_capacity_pct": drainage,
        "previous_24h_rainfall_mm": prev_24h,
        "soil_moisture_pct": soil_moisture,
        "elevation_m": elevation,
    }
    return derived, f"derived_from_{source}"


def get_models_status() -> dict[str, Any]:
    traffic_path = _resolve_model_path(settings.traffic_model_path)
    traffic_loaded = False
    traffic_error = None
    try:
        get_traffic_model()
        traffic_loaded = True
    except Exception as exc:
        traffic_error = str(exc)

    flood_path = _resolve_model_path(settings.flood_model_path)
    flood_loaded = False
    flood_error = None
    try:
        get_flood_model()
        flood_loaded = True
    except Exception as exc:
        flood_error = str(exc)
    flood_metrics = get_flood_metrics()

    return {
        "traffic_model": {
            "name": "Traffic Congestion Random Forest Regressor",
            "loaded": traffic_loaded,
            "artifact_exists": traffic_path.exists(),
            "artifact_path": str(traffic_path.relative_to(ROOT_DIR).as_posix()) if traffic_path.is_relative_to(ROOT_DIR) else str(traffic_path),
            "features": TRAFFIC_FEATURES,
            "training_data_type": "synthetic",
            "error": traffic_error,
        },
        "flood_model": {
            "name": "Flood Risk ML Classifier",
            "selected_algorithm": flood_metrics.get("selected_model", "unknown"),
            "loaded": flood_loaded,
            "artifact_exists": flood_path.exists(),
            "artifact_path": str(flood_path.relative_to(ROOT_DIR).as_posix()) if flood_path.is_relative_to(ROOT_DIR) else str(flood_path),
            "features": FLOOD_FEATURES,
            "evaluation_metrics": flood_metrics.get("selected_metrics", {}),
            "model_comparison": flood_metrics.get("model_comparison", []),
            "training_data_type": "synthetic",
            "error": flood_error,
        },
    }


def assess_risk(readings: dict) -> dict:
    """Transparent illustrative scoring only; not a calibrated safety model."""
    rain = float(readings["rainfall_mm_hour"])
    traffic = float(readings["traffic_density_pct"])
    water = float(readings["water_level_m"])
    aqi = float(readings["aqi"])
    drainage = float(readings["drainage_capacity_pct"])
    flood = 100 * (0.45 * _clamp(rain / 120) + 0.35 * _clamp(water / 2) + 0.20 * _clamp((100 - drainage) / 100))
    traffic_score = 100 * _clamp(traffic / 100)
    environment = 100 * _clamp(aqi / 300)
    overall = max(flood, traffic_score, environment)
    severity = "high" if overall >= 75 else "medium" if overall >= 45 else "low"
    return {
        "overall_score": round(overall, 1),
        "severity": severity,
        "domain_scores": {"flood": round(flood, 1), "traffic": round(traffic_score, 1), "environment": round(environment, 1)},
        "method": "illustrative_rule_based_demo",
        "limitation": "Not calibrated or validated for real-world safety decisions.",
    }
