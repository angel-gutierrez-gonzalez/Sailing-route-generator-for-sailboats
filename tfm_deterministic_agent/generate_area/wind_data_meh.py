import requests
from typing import List, Dict, Optional

def fetch_hourly_wind(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    base_url: str = "https://api.open-meteo.com/v1/forecast",
    windspeed_unit: str = "kn",
    timezone: str = "UTC",
) -> Optional[Dict[str, List]]:
    """
    Llama al servicio Open-Meteo para obtener datos horarios de viento en un punto.
    Devuelve un dict con:
      {"time": [...], "wind_speed_10m": [...], "wind_direction_10m": [...]} 
    o None si falla la petición.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "start_date": start_date,
        "end_date": end_date,
        "windspeed_unit": windspeed_unit,
        "timezone": timezone
    }
    try:
        response = requests.get(base_url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json().get("hourly", {})
        return {
            "time": data.get("time", []),
            "wind_speed_10m": data.get("wind_speed_10m", []),
            "wind_direction_10m": data.get("wind_direction_10m", [])
        }
    except requests.RequestException:
        return None


def evaluate_point_wind(
    lat: float,
    lon: float,
    depth_avg: Optional[float],
    boat_data: Dict,
    start_date: str,
    end_date: str,
    base_url: str = "https://api.open-meteo.com/v1/forecast",
    windspeed_unit: str = "kn",
    timezone: str = "UTC",
) -> List[Dict]:
    """
    Para un punto navegable, obtiene datos horarios de viento y marca la navegabilidad.
    Devuelve lista de dicts con:
      {latitude, longitude, depth_avg, time, wind_speed_10m, wind_direction_10m, navigable}
    """
    results = []
    wind_data = fetch_hourly_wind(lat, lon, start_date, end_date, base_url, windspeed_unit, timezone)
    if wind_data is None:
        # Fallback: un solo registro indicando fallo
        return [{
            "latitude": lat,
            "longitude": lon,
            "depth_avg": depth_avg,
            "time": None,
            "wind_speed_10m": None,
            "wind_direction_10m": None,
            "navigable": False
        }]

    times = wind_data["time"]
    speeds = wind_data["wind_speed_10m"]
    dirs = wind_data["wind_direction_10m"]

    for t, speed, direction in zip(times, speeds, dirs):
        navigable = speed is not None and speed <= boat_data.get("w_max", float('inf'))
        results.append({
            "latitude": lat,
            "longitude": lon,
            "depth_avg": depth_avg,
            "time": t,
            "wind_speed_10m": speed,
            "wind_direction_10m": direction,
            "navigable": navigable
        })
    return results


def batch_evaluate_wind(
    bathy_results: List[Dict],
    boat_data: Dict,
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """
    Para una lista de resultados de batimetría (con clave 'navigable'),
    obtiene datos de viento horarios solo para los puntos navegables.
    """
    all_results = []
    for item in bathy_results:
        if item.get("navigable"):
            lat = item["latitude"]
            lon = item["longitude"]
            depth_avg = item.get("depth_avg")
            wind_results = evaluate_point_wind(
                lat, lon, depth_avg, boat_data, start_date, end_date
            )
            all_results.extend(wind_results)
    return all_results
