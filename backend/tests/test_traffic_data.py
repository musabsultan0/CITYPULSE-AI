import httpx

from backend.app.core.config import settings
from backend.app.ml import predict_traffic_congestion
from backend.app.traffic_data import get_traffic_context_sync
from backend.app.workflow import run_workflow


def test_simulated_fallback_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "traffic_api_enabled", False)
    ctx = get_traffic_context_sync(zone="Zone A", use_live=True)
    assert ctx["is_simulated"] is True
    assert ctx["data_origin"] == "simulated_demo"
    assert ctx["live_api_attempted"] is False
    assert 0 <= ctx["metrics"]["congestion_pct"] <= 100
    assert "ml_features" in ctx


def test_simulated_fallback_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(settings, "traffic_api_enabled", True)
    monkeypatch.setattr(settings, "traffic_api_provider", "tomtom")
    monkeypatch.setattr(settings, "traffic_api_key", "")
    ctx = get_traffic_context_sync(zone="Zone B", use_live=True)
    assert ctx["is_simulated"] is True
    assert ctx["data_origin"] == "simulated_fallback"
    assert ctx["live_api_attempted"] is True
    assert "TRAFFIC_API_KEY" in ctx["fallback_reason"]


def test_live_tomtom_api_parsing_and_ml_integration(monkeypatch):
    monkeypatch.setattr(settings, "traffic_api_enabled", True)
    monkeypatch.setattr(settings, "traffic_api_provider", "tomtom")
    monkeypatch.setattr(settings, "traffic_api_key", "test-dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.tomtom.com" in str(request.url)
        return httpx.Response(
            200,
            json={
                "flowSegmentData": {
                    "frc": "FRC2",
                    "currentSpeed": 18,
                    "freeFlowSpeed": 60,
                    "currentTravelTime": 420,
                    "freeFlowTravelTime": 126,
                    "confidence": 0.96,
                    "roadClosure": False,
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as mock_client:
        ctx = get_traffic_context_sync(zone="Zone A", use_live=True, http_client=mock_client)

    assert ctx["is_simulated"] is False
    assert ctx["data_origin"] == "real_time_api"
    assert ctx["provider"] == "TomTom Traffic Flow API"
    assert ctx["metrics"]["current_speed_kmph"] == 18.0
    assert ctx["metrics"]["free_flow_speed_kmph"] == 60.0
    assert ctx["metrics"]["congestion_pct"] == 70.0
    assert ctx["metrics"]["delay_sec"] == 294

    ml_pred = predict_traffic_congestion(ctx["ml_features"], data_source=ctx["data_origin"], zone="Zone A")
    assert ml_pred["data_provenance"]["input_source"] == "real_time_api"
    assert 0 <= ml_pred["predicted_congestion_pct"] <= 100

    wf = run_workflow(
        {
            "scenario": "traffic_peak",
            "zone": "Zone A",
            "readings": {
                "rainfall_mm_hour": 0,
                "traffic_density_pct": ctx["metrics"]["congestion_pct"],
                "water_level_m": 0.1,
                "aqi": 80,
                "drainage_capacity_pct": 80,
                "source": "real_time_api",
            },
            "traffic_context": ctx,
        }
    )
    assert wf["traffic_finding"]["findings"]["data_origin"] == "real_time_api"


def test_live_here_api_parsing(monkeypatch):
    monkeypatch.setattr(settings, "traffic_api_enabled", True)
    monkeypatch.setattr(settings, "traffic_api_provider", "here")
    monkeypatch.setattr(settings, "traffic_api_key", "test-dummy-here-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "hereapi.com" in str(request.url)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "currentFlow": {
                            "speed": 10.0,
                            "freeFlow": 18.0,
                            "jamFactor": 6.5,
                            "confidence": 0.91,
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as mock_client:
        ctx = get_traffic_context_sync(zone="Zone C", use_live=True, http_client=mock_client)

    assert ctx["is_simulated"] is False
    assert ctx["data_origin"] == "real_time_api"
    assert ctx["provider"] == "HERE Traffic Flow API"
    assert ctx["metrics"]["congestion_pct"] == 65.0


def test_live_api_http_error_triggers_simulated_fallback(monkeypatch):
    monkeypatch.setattr(settings, "traffic_api_enabled", True)
    monkeypatch.setattr(settings, "traffic_api_provider", "tomtom")
    monkeypatch.setattr(settings, "traffic_api_key", "test-dummy-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "Service unavailable"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as mock_client:
        ctx = get_traffic_context_sync(zone="Zone A", use_live=True, http_client=mock_client)

    assert ctx["is_simulated"] is True
    assert ctx["data_origin"] == "simulated_fallback"
    assert ctx["live_api_attempted"] is True
    assert "HTTPStatusError" in ctx["fallback_reason"]
