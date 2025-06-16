# point_generation/generate_points.py
"""
Generación de malla de puntos dentro de un polígono dado.
"""
import math
from typing import List, Tuple
import numpy as np
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


def generate_points_in_area(
    area_shape: BaseGeometry,
    paso_lon: float
) -> List[Tuple[float, float]]:
    """
    Genera una malla regular de puntos dentro del polígono `area_shape` con separación `paso_lon` en grados.

    Parámetros:
        area_shape: polígono Shapely del área de estudio.
        paso_lon: espaciado en longitud (grados) para la malla.

    Retorna:
        Lista de tuplas (lat, lon) correspondientes a los nodos dentro del área.
    """
    # Bounding box del área
    minx, miny, maxx, maxy = area_shape.bounds

    # Número de intervalos en longitud
    N_lon = int(math.floor((maxx - minx) / paso_lon))
    lon_max_ajustada = minx + N_lon * paso_lon

    # Número de intervalos en latitud aproximando mismo paso
    N_lat = int(math.floor((maxy - miny) / paso_lon))
    lat_max_ajustada = miny + N_lat * paso_lon

    # Generar arrays de coordenadas
    lons = np.linspace(minx, lon_max_ajustada, N_lon + 1)
    lats = np.linspace(miny, lat_max_ajustada, N_lat + 1)

    # Filtrar puntos dentro del polígono
    puntos: List[Tuple[float, float]] = []
    for lat in lats:
        for lon in lons:
            if area_shape.contains(Point(lon, lat)):
                puntos.append((lat, lon))
    return puntos