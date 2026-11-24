from backend.app.main import demo_readings, models_status, predict_flood, predict_traffic
from backend.app.ml import assess_risk, predict_flood_risk, predict_traffic_congestion
from backend.app.schemas import FloodPredictionRequest, TrafficPredictionRequest
from backend.app.workflow import run_workflow


def test_risk_output_contract():
    result = assess_risk(
        {
            "rainfall_mm_hour": 85,
            "traffic_density_pct": 90,
            "water_level_m": 1.2,
            "aqi": 165,
            "drainage_capacity_pct": 45,
        }
    )
    assert result["severity"] in {"low", "medium", "high"}
    assert 0 <= result["overall_score"] <= 100
    assert result["method"] == "illustrative_rule_based_demo"


def test_traffic_random_forest_prediction():
    result = predict_traffic_congestion(
        {
            "hour": 18,
            "day_of_week": 2,
            "vehicle_count": 950,
            "average_speed": 25.0,
            "road_capacity": 600,
            "rainfall_mm": 12.0,
            "special_event": 1,
        }
    )
    assert result["model_info"]["model_available"] is True
    assert result["model_info"]["method"] == "random_forest_regressor"
    assert 70.0 <= result["predicted_congestion_pct"] <= 100.0
    assert result["traffic_status"] == "HIGH"
    assert result["data_provenance"]["is_synthetic_training_data"] is True


def test_flood_ml_prediction_high_and_low():
    high = predict_flood_risk(
        {
            "rainfall_mm_hour": 110.0,
            "water_level_m": 1.6,
            "drainage_capacity_pct": 30.0,
            "previous_24h_rainfall_mm": 160.0,
            "soil_moisture_pct": 85.0,
            "elevation_m": 8.0,
        }
    )
    assert high["model_info"]["model_available"] is True
    assert high["risk_level"] == "HIGH"
    assert high["flood_predicted"] is True
    assert 0.7 <= high["flood_probability"] <= 1.0
    assert high["data_provenance"]["is_synthetic_training_data"] is True

    low = predict_flood_risk(
        {
            "rainfall_mm_hour": 2.0,
            "water_level_m": 0.1,
            "drainage_capacity_pct": 90.0,
            "previous_24h_rainfall_mm": 5.0,
            "soil_moisture_pct": 25.0,
            "elevation_m": 75.0,
        }
    )
    assert low["risk_level"] == "LOW"
    assert low["flood_predicted"] is False
    assert low["flood_probability"] < 0.4


def test_models_status_and_endpoint_handlers():
    status = models_status()
    assert status["traffic_model"]["loaded"] is True
    assert status["flood_model"]["loaded"] is True
    assert status["flood_model"]["evaluation_metrics"]["roc_auc"] > 0.75

    t_res = predict_traffic(
        TrafficPredictionRequest(
            hour=18,
            day_of_week=2,
            vehicle_count=950,
            average_speed=25.0,
            road_capacity=600,
            rainfall_mm=12.0,
            special_event=1,
            zone="Zone A",
        )
    )
    assert t_res["traffic_status"] == "HIGH"

    f_res = predict_flood(
        FloodPredictionRequest(
            rainfall_mm_hour=95.0,
            water_level_m=1.4,
            drainage_capacity_pct=35.0,
            previous_24h_rainfall_mm=130.0,
            soil_moisture_pct=80.0,
            elevation_m=10.0,
            zone="Zone A",
        )
    )
    assert f_res["risk_level"] == "HIGH"


def test_workflow_integrates_both_ml_models():
    result = run_workflow(
        {
            "scenario": "heavy_rain",
            "zone": "Zone A",
            "readings": demo_readings("heavy_rain"),
        }
    )
    t_ml = result["traffic_finding"]["findings"]["ml_prediction"]
    f_ml = result["flood_finding"]["findings"]["ml_prediction"]
    assert t_ml["model_info"]["model_available"] is True
    assert f_ml["model_info"]["model_available"] is True
    assert 0 <= t_ml["predicted_congestion_pct"] <= 100
    assert 0 <= f_ml["flood_risk_pct"] <= 100
