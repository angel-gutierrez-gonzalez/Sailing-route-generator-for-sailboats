import math
import json
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from typing import Union
import os


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calcula la distancia en metros entre dos puntos geográficos usando la fórmula de Haversine.
    """
    R = 6371000  # Radio de la Tierra en metros
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calcula el rumbo (bearing) desde el punto 1 al punto 2 en grados [0,360).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    lam1, lam2 = math.radians(lon1), math.radians(lon2)
    y = math.sin(lam2 - lam1) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(lam2 - lam1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def load_single_area(area_geojson_path: str) -> BaseGeometry:
    """
    Carga un único archivo GeoJSON y devuelve su geometría (polígono o multipolígono).
    """
    if not os.path.isfile(area_geojson_path):
        raise FileNotFoundError(f"No se encontró el GeoJSON del área: {area_geojson_path}")
    with open(area_geojson_path, 'r', encoding='utf-8') as f:
        gj = json.load(f)
    # Asumimos que el GeoJSON contiene un FeatureCollection con al menos una feature
    geom = gj.get("features", [])[0].get("geometry")
    if geom is None:
        raise ValueError(f"El GeoJSON no contiene geometría en: {area_geojson_path}")
    return shape(geom)
