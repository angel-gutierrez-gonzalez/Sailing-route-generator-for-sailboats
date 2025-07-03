import math
import random
import numpy as np


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


def select_random_start_goal(df, max_distance_km=250.0, max_attempts=100):
    """
    Selecciona aleatoriamente un par de puntos (start, goal) navegables dentro de un DataFrame,
    asegurando que estén separados por menos de `max_distance_km` (Haversine recta, conservadora).

    Args:
        df (pd.DataFrame): Debe contener columnas ['latitude', 'longitude', 'navigable'].
        max_distance_km (float): Máxima distancia Haversine aceptable entre start y goal (en km).
        max_attempts (int): Número máximo de intentos antes de rendirse.

    Returns:
        (start, goal): Dos listas [lat, lon], o lanza ValueError si no hay combinación válida.
    """
    candidates = df[df['navigable']].copy()
    if len(candidates) < 2:
        raise ValueError("No hay suficientes puntos navegables para seleccionar start y goal")

    for _ in range(max_attempts):
        start_row = candidates.sample(1).iloc[0]
        goal_row = candidates.sample(1).iloc[0]

        if start_row.equals(goal_row):
            continue

        start = [start_row['latitude'], start_row['longitude']]
        goal = [goal_row['latitude'], goal_row['longitude']]

        distance = haversine(start, goal)
        if distance <= max_distance_km:
            return start, goal

    raise ValueError(f"No se encontró ningún par start-goal con distancia <= {max_distance_km} km tras {max_attempts} intentos")
