from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Any
import requests
from core.config import settings

class WeatherProvider(ABC):
    @abstractmethod
    def fetch_weather(self, lat: float, lon: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Fetch current weather and forecast."""
        pass

class OpenMeteoWeatherProvider(WeatherProvider):
    def fetch_weather(self, lat: float, lon: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        try:
            url = f"{settings.weather_api_url}?latitude={lat}&longitude={lon}&current_weather=true&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_max&timezone=auto"
            response = requests.get(url, timeout=settings.weather_timeout)
            response.raise_for_status()
            r = response.json()
            
            current_weather = {
                "temp": r['current_weather']['temperature'],
                "wind_speed": r['current_weather']['windspeed'],
                "weather_code": r['current_weather']['weathercode']
            }
            
            forecast_list = []
            daily_data = r.get('daily', {})
            temps_max = daily_data.get('temperature_2m_max', [])
            temps_min = daily_data.get('temperature_2m_min', [])
            rain_sum = daily_data.get('precipitation_sum', [])
            humidity = daily_data.get('relative_humidity_2m_max', [])
            times = daily_data.get('time', [])
            
            for i in range(1, 4):
                if i < len(temps_max):
                    forecast_list.append({
                        "day": i,
                        "date": times[i] if i < len(times) else "N/A",
                        "temp_max": temps_max[i] if i < len(temps_max) else "N/A",
                        "temp_min": temps_min[i] if i < len(temps_min) else "N/A",
                        "rain_mm": rain_sum[i] if i < len(rain_sum) else 0,
                        "humidity": humidity[i] if i < len(humidity) else "N/A"
                    })
            
            return current_weather, forecast_list
            
        except requests.exceptions.Timeout:
            print(f"[!] Weather API timeout (>{settings.weather_timeout}s). Check internet connection.")
            return {"temp": "N/A", "wind_speed": "N/A", "weather_code": "N/A"}, []
        except Exception as e:
            print(f"[!] Error fetching weather: {e}")
            return {"temp": "N/A", "wind_speed": "N/A", "weather_code": "N/A"}, []
