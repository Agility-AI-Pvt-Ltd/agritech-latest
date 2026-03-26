"""
pipeline/tools/weather.py  –  Geocoding + weather fetch tools.
"""
from __future__ import annotations

from typing import Any, Dict

import requests


def execute_geocode_location(address: str) -> Dict[str, Any]:
    """Resolve an address to latitude and longitude using Geopy/Nominatim."""
    from geopy.geocoders import Nominatim
    try:
        geolocator = Nominatim(user_agent="agritech_ai_assistant")
        location = geolocator.geocode(address)
        if location:
            return {
                "address":          address,
                "resolved_address": location.address,
                "latitude":         location.latitude,
                "longitude":        location.longitude,
            }
        return {
            "error": (
                f"Could not resolve the address: '{address}'. "
                "Please ask the user for a more specific city or village name."
            )
        }
    except Exception as e:
        return {"error": str(e)}


def execute_get_weather(
    latitude: float = None,
    longitude: float = None,
    location_name: str = None,
) -> Dict[str, Any]:
    """Fetch current weather and 3-day forecast from Open-Meteo.

    Accepts either lat/lon OR a location_name (auto-geocodes if name given).
    """
    try:
        # 1. Geocoding fallback: if name provided but coords missing
        if location_name and (latitude is None or longitude is None):
            geo = execute_geocode_location(location_name)
            if "error" in geo:
                return geo
            latitude  = geo["latitude"]
            longitude = geo["longitude"]

        # 2. Validation
        if latitude is None or longitude is None:
            return {
                "error": (
                    "Missing coordinates. Please provide 'latitude' and 'longitude' "
                    "OR a 'location_name'."
                )
            }

        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            f"&current_weather=true"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"relative_humidity_2m_max&timezone=auto"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()

        data  = r.json()
        cw    = data.get("current_weather", {})
        daily = data.get("daily", {})

        forecast = []
        times = daily.get("time", [])
        for i in range(1, 4):
            if i < len(times):
                forecast.append({
                    "date":     times[i],
                    "temp_max": daily.get("temperature_2m_max",       [None] * 5)[i],
                    "temp_min": daily.get("temperature_2m_min",       [None] * 5)[i],
                    "rain_mm":  daily.get("precipitation_sum",        [0]    * 5)[i],
                    "humidity": daily.get("relative_humidity_2m_max", [None] * 5)[i],
                })

        return {
            "current": {
                "temperature":  cw.get("temperature"),
                "wind_speed":   cw.get("windspeed"),
                "weather_code": cw.get("weathercode"),
            },
            "forecast_3day": forecast,
        }
    except Exception as e:
        return {"error": str(e)}
