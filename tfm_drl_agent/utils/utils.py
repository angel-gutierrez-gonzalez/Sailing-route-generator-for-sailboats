import math
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
    return (2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))) / 1000


def bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calcula el rumbo (bearing) desde el punto 1 al punto 2 en grados [0,360).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    lam1, lam2 = math.radians(lon1), math.radians(lon2)
    y = math.sin(lam2 - lam1) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(lam2 - lam1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def select_random_start_goal(df, max_distance_km=220.0):
    """
    Selecciona un par (start, goal) aleatorio pero asegurando que estén a una distancia navegable.
    Aplica un filtro geográfico previo (bounding box) para reducir el cálculo de distancias.
    """
    # Reducir a ubicaciones únicas
    df_unique = df[['latitude', 'longitude', 'navigable']].drop_duplicates()
    valid_rows = df_unique[df_unique['navigable']].reset_index(drop=True)

    if len(valid_rows) < 2:
        raise ValueError("No hay suficientes puntos navegables")

    for _ in range(1000):
        start_row = valid_rows.sample(1).iloc[0]
        start = [start_row["latitude"], start_row["longitude"]]

        # Definir límites aproximados en grados (1 grado ≈ 111 km)
        lat_tol = max_distance_km / 111.0
        lon_tol = max_distance_km / 111.0

        lat0, lon0 = start
        subset = valid_rows[
            (valid_rows['latitude'] >= lat0 - lat_tol) &
            (valid_rows['latitude'] <= lat0 + lat_tol) &
            (valid_rows['longitude'] >= lon0 - lon_tol) &
            (valid_rows['longitude'] <= lon0 + lon_tol) &
            ((valid_rows['latitude'] != lat0) | (valid_rows['longitude'] != lon0))
        ].copy()

        if not subset.empty:
            goal_row = subset.sample(1).iloc[0]
            goal = [goal_row["latitude"], goal_row["longitude"]]
            return start, goal

    raise ValueError("No se encontró ningún par start-goal válido dentro del límite de distancia.")

def estimate_heel(relative_wind_angle, boat_speed):
    """
    Estima la escora (heel) del barco en grados, como función del ángulo de viento relativo y la velocidad del barco.
    Se asume una aproximación simplificada:
    - Máxima escora a 90° de viento aparente (beam reach)
    - Cero escora con viento de proa o popa (0° o 180°)
    """
    if relative_wind_angle > 180:
        relative_wind_angle = 360 - relative_wind_angle  # simetría

    factor = np.sin(np.radians(relative_wind_angle))
    heel = factor * boat_speed / 2  # relación arbitraria para escora máxima ~20°
    return np.clip(heel, 0, 25)  # limitar escora máxima realista
