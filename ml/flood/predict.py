from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flood_model.joblib"

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame([{
    "rainfall_mm_hour": 85.0,
    "water_level_m": 1.35,
    "drainage_capacity_pct": 40.0,
    "previous_24h_rainfall_mm": 120.0,
    "soil_moisture_pct": 78.0,
    "elevation_m": 12.0,
}])

prob = float(model.predict_proba(sample)[0, 1])
risk_pct = max(0.0, min(100.0, prob * 100.0))

if risk_pct >= 70:
    status = "HIGH"
elif risk_pct >= 40:
    status = "MODERATE"
else:
    status = "LOW"

print("\n====================================")
print("      CITYPULSE AI - FLOOD ML")
print("====================================")
print(f"Predicted Flood Probability: {risk_pct:.2f}%")
print(f"Flood Risk Level: {status}")
print("Note: Trained on synthetic data only.")
print("====================================")
