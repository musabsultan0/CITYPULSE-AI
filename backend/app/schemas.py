from typing import Literal, Any
from pydantic import BaseModel, Field

class SensorReadings(BaseModel):
    rainfall_mm_hour: float = Field(ge=0, le=500)
    traffic_density_pct: float = Field(ge=0, le=100)
    water_level_m: float = Field(ge=0, le=20)
    aqi: float = Field(ge=0, le=1000)
    drainage_capacity_pct: float = Field(ge=0, le=100, default=60)
    source: str = "simulated_demo"

class TrafficPredictionRequest(BaseModel):
    hour: int = Field(default=18, ge=0, le=23)
    day_of_week: int = Field(default=2, ge=0, le=6)
    vehicle_count: float = Field(default=750, ge=0, le=5000)
    average_speed: float = Field(default=30.0, ge=0, le=150)
    road_capacity: float = Field(default=600, gt=0, le=5000)
    rainfall_mm: float = Field(default=5.0, ge=0, le=500)
    special_event: int = Field(default=0, ge=0, le=1)
    zone: str = Field(default="Zone A", min_length=1, max_length=100)
    data_source: str = Field(default="user_supplied", max_length=100)

class FloodPredictionRequest(BaseModel):
    rainfall_mm_hour: float = Field(default=65.0, ge=0, le=500)
    water_level_m: float = Field(default=1.1, ge=0, le=20)
    drainage_capacity_pct: float = Field(default=45.0, ge=0, le=100)
    previous_24h_rainfall_mm: float = Field(default=95.0, ge=0, le=1000)
    soil_moisture_pct: float = Field(default=72.0, ge=0, le=100)
    elevation_m: float = Field(default=15.0, ge=0, le=5000)
    zone: str = Field(default="Zone A", min_length=1, max_length=100)
    data_source: str = Field(default="user_supplied", max_length=100)

class AnalyzeRequest(BaseModel):
    scenario: Literal["heavy_rain", "traffic_peak", "poor_air_quality", "normal", "custom"] = "heavy_rain"
    zone: str = Field(default="Zone A", min_length=1, max_length=100)
    readings: SensorReadings | None = None
    traffic_features: TrafficPredictionRequest | None = None
    flood_features: FloodPredictionRequest | None = None
    use_live_weather: bool = False
    use_live_traffic: bool = False
    use_llm: bool | None = None

class ApprovalRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=2, max_length=100)
    comment: str = Field(default="", max_length=1000)

class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
