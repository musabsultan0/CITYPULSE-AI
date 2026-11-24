from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
from typing import Any

import httpx

from backend.app.core.config import settings

# Zone coordinate offsets around the configured city center (default: Hyderabad)
ZONE_OFFSETS: dict[str, tuple[float, float]] = {
    "Zone A": (0.0, 0.0),
    "Zone B": (0.032, 0.025),
    "Zone C": (-0.028, 0.041),
}

ZONE_ROAD_CAPACITY: dict[str, float] = {
    "Zone A": 650.0,
    "Zone B": 800.0,
    "Zone C": 550.0,
}


def resolve_zone_coordinates(
    zone: str = "Zone A",
    latitude: float | None = None,
    longitude: float | None = None,
) -> tuple[float, float]:
    if latitude is not None and longitude is not None:
        return round(float(latitude), 6), round(float(longitude), 6)
    dlat, dlon = ZONE_OFFSETS.get(zone, (0.0, 0.0))
    return round(settings.latitude + dlat, 6), round(settings.longitude + dlon, 6)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _build_ml_features_from_flow(
    zone: str,
    current_speed_kmph: float,
    free_flow_speed_kmph: float,
    congestion_pct: float,
    rainfall_mm: float = 0.0,
    special_event: int = 0,
    data_source: str = "simulated_fallback",
    now: datetime | None = None,
) -> dict[str, Any]:
    ts = now or datetime.now()
    road_capacity = ZONE_ROAD_CAPACITY.get(zone, 600.0)
    speed_ratio = _clamp(current_speed_kmph / max(free_flow_speed_kmph, 1.0), 0.1, 1.2)
    estimated_load = _clamp(0.25 + 1.35 * (congestion_pct / 100.0) + 0.2 * (1.0 - speed_ratio), 0.1, 2.0)
    vehicle_count = round(road_capacity * estimated_load, 1)
    return {
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "vehicle_count": vehicle_count,
        "average_speed": round(current_speed_kmph, 2),
        "road_capacity": road_capacity,
        "rainfall_mm": round(max(0.0, rainfall_mm), 2),
        "special_event": int(special_event),
        "zone": zone,
        "data_source": data_source,
    }


def build_simulated_traffic_context(
    zone: str = "Zone A",
    latitude: float | None = None,
    longitude: float | None = None,
    rainfall_mm: float = 0.0,
    special_event: int = 0,
    fallback_reason: str = "Live traffic API disabled; simulated fallback active.",
    live_api_attempted: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Deterministic, time-of-day aware simulated traffic feed with explicit fallback provenance."""
    ts = now or datetime.now(timezone.utc)
    lat, lon = resolve_zone_coordinates(zone, latitude, longitude)
    hour = ts.hour

    # Time-of-day realistic urban traffic curve (morning + evening peaks)
    morning_peak = math.exp(-((hour - 8.5) ** 2) / 5.0)
    evening_peak = math.exp(-((hour - 18.0) ** 2) / 6.0)
    zone_seed = int(hashlib.sha256(f"{zone}:{ts.date().isoformat()}:{hour}".encode()).hexdigest()[:6], 16)
    zone_jitter = ((zone_seed % 15) - 7)  # -7 to +7

    free_flow_speed = 62.0 if zone == "Zone B" else (55.0 if zone == "Zone A" else 50.0)
    congestion = (
        22.0
        + 46.0 * max(morning_peak, evening_peak)
        + 0.45 * min(rainfall_mm, 50.0)
        + (14.0 if special_event else 0.0)
        + zone_jitter
    )
    congestion = round(_clamp(congestion, 8.0, 96.0), 2)
    current_speed = round(max(8.0, free_flow_speed * (1.0 - 0.78 * (congestion / 100.0))), 2)
    free_flow_tt = 300  # seconds over nominal corridor segment
    current_tt = int(round(free_flow_tt * (free_flow_speed / max(current_speed, 5.0))))

    origin = "simulated_fallback" if live_api_attempted else "simulated_demo"
    ml_features = _build_ml_features_from_flow(
        zone=zone,
        current_speed_kmph=current_speed,
        free_flow_speed_kmph=free_flow_speed,
        congestion_pct=congestion,
        rainfall_mm=rainfall_mm,
        special_event=special_event,
        data_source=origin,
        now=ts,
    )

    return {
        "zone": zone,
        "provider": "Simulated Urban Traffic Generator",
        "data_origin": origin,
        "is_simulated": True,
        "live_api_attempted": live_api_attempted,
        "fallback_reason": fallback_reason,
        "timestamp_utc": ts.isoformat(),
        "coordinates": {"latitude": lat, "longitude": lon},
        "metrics": {
            "current_speed_kmph": current_speed,
            "free_flow_speed_kmph": free_flow_speed,
            "congestion_pct": congestion,
            "current_travel_time_sec": current_tt,
            "free_flow_travel_time_sec": free_flow_tt,
            "delay_sec": max(0, current_tt - free_flow_tt),
            "road_closure": False,
            "confidence": 0.0,
        },
        "ml_features": ml_features,
        "note": "Simulated traffic telemetry (not live road sensor or traffic API data).",
    }


def _parse_tomtom_response(
    payload: dict[str, Any],
    zone: str,
    lat: float,
    lon: float,
    rainfall_mm: float,
    special_event: int,
) -> dict[str, Any]:
    seg = payload.get("flowSegmentData")
    if not seg:
        raise ValueError("TomTom response missing 'flowSegmentData'")
    current_speed = float(seg["currentSpeed"])
    free_flow_speed = float(seg.get("freeFlowSpeed") or max(current_speed, 1.0))
    current_tt = int(seg.get("currentTravelTime", 0))
    free_flow_tt = int(seg.get("freeFlowTravelTime", 0))
    confidence = float(seg.get("confidence", 0.0))
    road_closure = bool(seg.get("roadClosure", False))
    congestion = round(_clamp((1.0 - (current_speed / max(free_flow_speed, 1.0))) * 100.0, 0.0, 100.0), 2)
    if road_closure:
        congestion = 100.0

    ts = datetime.now(timezone.utc)
    ml_features = _build_ml_features_from_flow(
        zone=zone,
        current_speed_kmph=current_speed,
        free_flow_speed_kmph=free_flow_speed,
        congestion_pct=congestion,
        rainfall_mm=rainfall_mm,
        special_event=special_event,
        data_source="real_time_api",
        now=ts,
    )
    return {
        "zone": zone,
        "provider": "TomTom Traffic Flow API",
        "data_origin": "real_time_api",
        "is_simulated": False,
        "live_api_attempted": True,
        "fallback_reason": None,
        "timestamp_utc": ts.isoformat(),
        "coordinates": {"latitude": lat, "longitude": lon},
        "metrics": {
            "current_speed_kmph": round(current_speed, 2),
            "free_flow_speed_kmph": round(free_flow_speed, 2),
            "congestion_pct": congestion,
            "current_travel_time_sec": current_tt,
            "free_flow_travel_time_sec": free_flow_tt,
            "delay_sec": max(0, current_tt - free_flow_tt),
            "road_closure": road_closure,
            "confidence": round(confidence, 3),
            "functional_road_class": seg.get("frc"),
        },
        "ml_features": ml_features,
        "note": "Live road-segment speed telemetry fetched from TomTom Traffic Flow API.",
    }


def _parse_here_response(
    payload: dict[str, Any],
    zone: str,
    lat: float,
    lon: float,
    rainfall_mm: float,
    special_event: int,
) -> dict[str, Any]:
    results = payload.get("results") or []
    if not results or "currentFlow" not in results[0]:
        raise ValueError("HERE response missing 'results[0].currentFlow'")
    flow = results[0]["currentFlow"]
    # HERE v7 returns speeds in m/s
    current_speed = float(flow.get("speed", 0.0)) * 3.6
    free_flow_speed = float(flow.get("freeFlow", max(current_speed / 3.6, 1.0))) * 3.6
    jam_factor = float(flow.get("jamFactor", 0.0))
    congestion = round(_clamp(jam_factor * 10.0, 0.0, 100.0), 2)
    confidence = float(flow.get("confidence", 0.0))

    ts = datetime.now(timezone.utc)
    ml_features = _build_ml_features_from_flow(
        zone=zone,
        current_speed_kmph=current_speed,
        free_flow_speed_kmph=free_flow_speed,
        congestion_pct=congestion,
        rainfall_mm=rainfall_mm,
        special_event=special_event,
        data_source="real_time_api",
        now=ts,
    )
    return {
        "zone": zone,
        "provider": "HERE Traffic Flow API",
        "data_origin": "real_time_api",
        "is_simulated": False,
        "live_api_attempted": True,
        "fallback_reason": None,
        "timestamp_utc": ts.isoformat(),
        "coordinates": {"latitude": lat, "longitude": lon},
        "metrics": {
            "current_speed_kmph": round(current_speed, 2),
            "free_flow_speed_kmph": round(free_flow_speed, 2),
            "congestion_pct": congestion,
            "jam_factor": round(jam_factor, 2),
            "road_closure": jam_factor >= 10.0,
            "confidence": round(confidence, 3),
        },
        "ml_features": ml_features,
        "note": "Live traffic flow telemetry fetched from HERE Traffic API v7.",
    }


def _build_provider_request(provider: str, lat: float, lon: float, api_key: str) -> tuple[str, dict[str, Any]]:
    provider_clean = provider.lower().strip()
    if provider_clean == "tomtom":
        return (
            "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
            {"point": f"{lat},{lon}", "unit": "KMPH", "key": api_key},
        )
    if provider_clean == "here":
        return (
            "https://data.traffic.hereapi.com/v7/flow",
            {"in": f"circle:{lat},{lon};r=500", "locationReferencing": "none", "apiKey": api_key},
        )
    raise ValueError(f"Unsupported TRAFFIC_API_PROVIDER: '{provider}'")


def get_traffic_context_sync(
    zone: str = "Zone A",
    use_live: bool = True,
    latitude: float | None = None,
    longitude: float | None = None,
    rainfall_mm: float = 0.0,
    special_event: int = 0,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Synchronous live traffic fetch with guaranteed graceful simulated fallback."""
    lat, lon = resolve_zone_coordinates(zone, latitude, longitude)
    provider = settings.traffic_api_provider.lower().strip()

    if not use_live or not settings.traffic_api_enabled or provider == "simulated":
        reason = (
            "Live traffic disabled by request."
            if not use_live
            else "TRAFFIC_API_ENABLED is false (configure TRAFFIC_API_ENABLED=true and TRAFFIC_API_KEY in .env)."
            if not settings.traffic_api_enabled
            else "TRAFFIC_API_PROVIDER set to simulated."
        )
        return build_simulated_traffic_context(
            zone=zone,
            latitude=lat,
            longitude=lon,
            rainfall_mm=rainfall_mm,
            special_event=special_event,
            fallback_reason=reason,
            live_api_attempted=False,
        )

    api_key = settings.traffic_api_key.strip()
    if not api_key:
        return build_simulated_traffic_context(
            zone=zone,
            latitude=lat,
            longitude=lon,
            rainfall_mm=rainfall_mm,
            special_event=special_event,
            fallback_reason="TRAFFIC_API_KEY is not configured in environment variables; fell back to simulated traffic.",
            live_api_attempted=True,
        )

    try:
        url, params = _build_provider_request(provider, lat, lon, api_key)
        if http_client is not None:
            resp = http_client.get(url, params=params, timeout=settings.traffic_api_timeout_seconds)
        else:
            with httpx.Client(timeout=settings.traffic_api_timeout_seconds) as client:
                resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if provider == "tomtom":
            return _parse_tomtom_response(payload, zone, lat, lon, rainfall_mm, special_event)
        return _parse_here_response(payload, zone, lat, lon, rainfall_mm, special_event)
    except Exception as exc:
        return build_simulated_traffic_context(
            zone=zone,
            latitude=lat,
            longitude=lon,
            rainfall_mm=rainfall_mm,
            special_event=special_event,
            fallback_reason=f"Live {provider} traffic API error ({type(exc).__name__}: {exc}); fell back to simulated traffic.",
            live_api_attempted=True,
        )


async def get_traffic_context(
    zone: str = "Zone A",
    use_live: bool = True,
    latitude: float | None = None,
    longitude: float | None = None,
    rainfall_mm: float = 0.0,
    special_event: int = 0,
) -> dict[str, Any]:
    """Async live traffic fetch for FastAPI routes with automatic simulated fallback."""
    lat, lon = resolve_zone_coordinates(zone, latitude, longitude)
    provider = settings.traffic_api_provider.lower().strip()

    if not use_live or not settings.traffic_api_enabled or provider == "simulated":
        return get_traffic_context_sync(
            zone=zone,
            use_live=use_live,
            latitude=lat,
            longitude=lon,
            rainfall_mm=rainfall_mm,
            special_event=special_event,
        )

    api_key = settings.traffic_api_key.strip()
    if not api_key:
        return build_simulated_traffic_context(
            zone=zone,
            latitude=lat,
            longitude=lon,
            rainfall_mm=rainfall_mm,
            special_event=special_event,
            fallback_reason="TRAFFIC_API_KEY is not configured in environment variables; fell back to simulated traffic.",
            live_api_attempted=True,
        )

    try:
        url, params = _build_provider_request(provider, lat, lon, api_key)
        async with httpx.AsyncClient(timeout=settings.traffic_api_timeout_seconds) as client:
            resp = await client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if provider == "tomtom":
            return _parse_tomtom_response(payload, zone, lat, lon, rainfall_mm, special_event)
        return _parse_here_response(payload, zone, lat, lon, rainfall_mm, special_event)
    except Exception as exc:
        return build_simulated_traffic_context(
            zone=zone,
            latitude=lat,
            longitude=lon,
            rainfall_mm=rainfall_mm,
            special_event=special_event,
            fallback_reason=f"Live {provider} traffic API error ({type(exc).__name__}: {exc}); fell back to simulated traffic.",
            live_api_attempted=True,
        )
