
from pathlib import Path
import numpy as np
import pandas as pd

# Reproducible synthetic data
rng = np.random.default_rng(42)
n = 5000

# Generate input features
hour = rng.integers(0, 24, n)
day_of_week = rng.integers(0, 7, n)
vehicle_count = rng.integers(50, 1201, n)
average_speed = rng.uniform(10, 80, n)
road_capacity = rng.integers(300, 1001, n)
rainfall_mm = rng.uniform(0, 30, n)
special_event = rng.integers(0, 2, n)

# Identify peak traffic hours
peak_hour = np.isin(
    hour, [7, 8, 9, 17, 18, 19]
).astype(int)

# Calculate illustrative congestion percentage
traffic_load = vehicle_count / road_capacity

congestion_pct = (
    35 * np.clip(traffic_load, 0, 2)
    + 20 * (1 - average_speed / 80)
    + 10 * (rainfall_mm / 30)
    + 12 * peak_hour
    + 8 * special_event
    + rng.normal(0, 5, n)
)

congestion_pct = np.clip(congestion_pct, 0, 100)

# Create dataset
df = pd.DataFrame({
    "hour": hour,
    "day_of_week": day_of_week,
    "vehicle_count": vehicle_count,
    "average_speed": average_speed,
    "road_capacity": road_capacity,
    "rainfall_mm": rainfall_mm,
    "special_event": special_event,
    "congestion_pct": congestion_pct
})

# Save dataset
output_path = (
    Path(__file__).resolve().parent
    / "datasets"
    / "traffic_data.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("Traffic dataset created successfully!")
print("Dataset shape:", df.shape)
print("Saved to:", output_path)
print("\nFirst five records:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
