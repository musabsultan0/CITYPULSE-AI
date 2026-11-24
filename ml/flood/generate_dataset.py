from pathlib import Path
import numpy as np
import pandas as pd

# Reproducible synthetic flood dataset for CITYPULSE AI prototype
# NOTE: Synthetic demonstration data only; not calibrated on historical municipal flood records.
rng = np.random.default_rng(42)
n = 6000

# Realistic urban meteorological & hydrological feature distributions
rainfall_mm_hour = np.clip(rng.gamma(shape=1.6, scale=22.0, size=n), 0.0, 200.0)
previous_24h_rainfall_mm = np.clip(
    rainfall_mm_hour * rng.uniform(0.8, 2.2, size=n) + rng.gamma(shape=1.5, scale=20.0, size=n),
    0.0,
    300.0,
)
drainage_capacity_pct = rng.uniform(10.0, 100.0, size=n)
elevation_m = np.clip(rng.gamma(shape=2.0, scale=18.0, size=n), 0.5, 120.0)
soil_moisture_pct = np.clip(
    25.0 + 0.22 * previous_24h_rainfall_mm + rng.normal(0, 12.0, size=n),
    10.0,
    100.0,
)
water_level_m = np.clip(
    0.15
    + 0.011 * rainfall_mm_hour
    + 0.0025 * previous_24h_rainfall_mm
    + 0.45 * (1.0 - drainage_capacity_pct / 100.0)
    + 0.35 * np.exp(-elevation_m / 25.0)
    + rng.normal(0, 0.28, size=n),
    0.0,
    3.5,
)

# Nonlinear flood risk latent score combining hydrological drivers
low_elevation_factor = np.exp(-elevation_m / 30.0)
drainage_deficit = 1.0 - (drainage_capacity_pct / 100.0)
rain_drainage_interaction = (rainfall_mm_hour / 150.0) * drainage_deficit

latent_risk = (
    0.34 * np.clip(rainfall_mm_hour / 140.0, 0.0, 1.5)
    + 0.28 * np.clip(water_level_m / 2.2, 0.0, 1.5)
    + 0.16 * drainage_deficit
    + 0.10 * np.clip(previous_24h_rainfall_mm / 220.0, 0.0, 1.5)
    + 0.08 * (soil_moisture_pct / 100.0)
    + 0.12 * low_elevation_factor
    + 0.18 * rain_drainage_interaction
    + rng.normal(0, 0.045, size=n)
)

flood_prob = 1.0 / (1.0 + np.exp(-11.0 * (latent_risk - 0.54)))
flood_occurred = rng.binomial(1, flood_prob)

df = pd.DataFrame({
    "rainfall_mm_hour": np.round(rainfall_mm_hour, 2),
    "water_level_m": np.round(water_level_m, 3),
    "drainage_capacity_pct": np.round(drainage_capacity_pct, 2),
    "previous_24h_rainfall_mm": np.round(previous_24h_rainfall_mm, 2),
    "soil_moisture_pct": np.round(soil_moisture_pct, 2),
    "elevation_m": np.round(elevation_m, 2),
    "flood_occurred": flood_occurred.astype(int),
})

output_path = Path(__file__).resolve().parent / "datasets" / "flood_data.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("Synthetic flood dataset created successfully!")
print("Dataset shape:", df.shape)
print(f"Positive flood rate: {df['flood_occurred'].mean():.2%}")
print("Saved to:", output_path)
print("\nFirst five records:")
print(df.head())
