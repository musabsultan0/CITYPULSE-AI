import httpx
from backend.app.core.config import settings

async def get_weather_context():
    if not settings.open_meteo_enabled:
        return {"enabled":False,"message":"Live weather disabled; supplied/demo readings used."}
    params={"latitude":settings.latitude,"longitude":settings.longitude,"hourly":"precipitation,precipitation_probability","forecast_days":1,"timezone":"auto"}
    async with httpx.AsyncClient(timeout=15) as client:
        response=await client.get("https://api.open-meteo.com/v1/forecast",params=params)
        response.raise_for_status()
        data=response.json()
    hourly=data.get("hourly",{})
    return {"enabled":True,"source":"Open-Meteo","hourly":{"time":hourly.get("time",[])[:6],"precipitation_mm":hourly.get("precipitation",[])[:6],"precipitation_probability_pct":hourly.get("precipitation_probability",[])[:6]},"caveat":"Forecast is not a street-level flood measurement."}
