import requests
from typing import List, Tuple, Optional, Dict

def fetch_depth(lat: float, lon: float, timeout: int = 10) -> Optional[float]:
    """
    Llama al servicio EMODnet para obtener la profundidad media en un punto.
    Devuelve None si falla la petición o no hay dato.
    """
    wkt = f"POINT({lon} {lat})"
    url = f"https://rest.emodnet-bathymetry.eu/depth_sample?geom={wkt}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("avg", None)
    except requests.RequestException:
        return None


def evaluate_navigability(
    lat: float,
    lon: float,
    draft: float
) -> Dict[str, Optional[float or bool]]:
    """
    Para un punto (lat, lon) y un draft dado, obtiene la profundidad
    y calcula si es navegable:
      - profundidad < 0 (bajo nivel del mar)
      - |profundidad| >= draft
    Devuelve un dict con:
      {
        "latitude": lat,
        "longitude": lon,
        "depth_avg": profundidad_media,
        "navigable": bool or None
      }
    """
    depth = fetch_depth(lat, lon)
    navigable = None
    if depth is not None:
        navigable = (depth < 0) and (abs(depth) >= draft)
    return {
        "latitude": lat,
        "longitude": lon,
        "depth_avg": depth,
        "navigable": navigable
    }


def batch_evaluate_navigability(
    points: List[Tuple[float, float]],
    draft: float
) -> List[Dict[str, Optional[float or bool]]]:
    """
    Para una lista de puntos [(lat, lon), ...] y un draft,
    devuelve una lista de resultados con profundidad y navegabilidad.
    """
    resultados = []
    for lat, lon in points:
        resultados.append(evaluate_navigability(lat, lon, draft))
    return resultados
