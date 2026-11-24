
from pathlib import Path
import joblib
import pandas as pd

# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
MODEL_PATH = BASE_DIR / "models" / "traffic_model.joblib"

model = joblib.load(MODEL_PATH)

# Input: New traffic conditions
new_data = pd.DataFrame([{
    "hour": 18,
    "day_of_week": 2,
    "vehicle_count": 950,
    "average_speed": 25.0,
    "road_capacity": 600,
    "rainfall_mm": 12.0,
    "special_event": 1
}])

# Generate prediction
prediction = model.predict(new_data)[0]

# Keep prediction within 0-100
prediction = max(0, min(100, prediction))

# Display results
print("\n====================================")
print("     CITYPULSE AI - TRAFFIC")
print("====================================")

print(f"Predicted Congestion: {prediction:.2f}%")

if prediction >= 70:
    status = "HIGH"
elif prediction >= 40:
    status = "MODERATE"
else:
    status = "LOW"

print(f"Traffic Status: {status}")
print("====================================")
