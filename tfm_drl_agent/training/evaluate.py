import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from utils.utils import haversine, estimate_heel

def analizar_ruta(df):
    """
    Analiza una ruta y devuelve métricas como distancia, escora, viradas, etc.
    Usa la columna 'heel' si está disponible, si no la calcula a partir del viento y el rumbo.
    """

    total_dist = 0.0
    heading_changes = []
    heel_segments = []
    current_heel = None
    duration = 0

    use_heel_column = 'heel' in df.columns

    for i in range(1, len(df)):
        total_dist += haversine(df.iloc[i - 1]['lon'], df.iloc[i - 1]['lat'], df.iloc[i]['lon'], df.iloc[i]['lat'])

        # Cambios de rumbo
        heading_diff = abs(df.iloc[i]['heading'] - df.iloc[i - 1]['heading'])
        if heading_diff > 10:
            heading_changes.append(i)

        # Escora
        if use_heel_column:
            heel = df.iloc[i]['heel']
        elif all(col in df.columns for col in ['wind_dir', 'heading', 'speed']):
            rel_angle = (df.iloc[i]['wind_dir'] - df.iloc[i]['heading']) % 360
            heel = estimate_heel(rel_angle, df.iloc[i]['speed'])
        else:
            heel = None

        if heel is not None:
            if current_heel is None:
                current_heel = heel
                duration = 1
            elif np.isclose(heel, current_heel, atol=0.1):
                duration += 1
            else:
                heel_segments.append((current_heel, duration))
                current_heel = heel
                duration = 1

    if current_heel is not None:
        heel_segments.append((current_heel, duration))

    if not heel_segments:
        return {
            'distancia_km': total_dist,
            'duracion_steps': len(df),
            'heel_max': None,
            'heel_min': None,
            'heel_prom': None,
            'viradas': len(heading_changes)
        }

    max_heel, max_duration = max(heel_segments, key=lambda x: x[0])
    min_heel, min_duration = min(heel_segments, key=lambda x: x[0])
    prom_heel = np.mean([h for h, _ in heel_segments])

    return {
        'distancia_km': total_dist,
        'duracion_steps': len(df),
        'heel_max': max_heel,
        'heel_max_duracion': max_duration,
        'heel_min': min_heel,
        'heel_min_duracion': min_duration,
        'heel_prom': prom_heel,
        'viradas': len(heading_changes)
    }


def comparar_rutas_detallado(drl_path, det_path):
    df_drl = pd.read_csv(drl_path)
    df_det = pd.read_csv(det_path)

    # Normalizar columnas
    rename_cols = {
        'wind_direction_10m': 'wind_dir',
        'wind_speed_10m': 'wind_speed',
        'latitude': 'lat',
        'longitude': 'lon'
    }
    df_drl.rename(columns={k: v for k, v in rename_cols.items() if k in df_drl.columns}, inplace=True)
    df_det.rename(columns={k: v for k, v in rename_cols.items() if k in df_det.columns}, inplace=True)

    analisis_drl = analizar_ruta(df_drl)
    analisis_det = analizar_ruta(df_det)

    comparacion = {
        'drl': analisis_drl,
        'det': analisis_det
    }
    return comparacion

def mostrar_rutas_en_mapa_folium(path_drl, path_det):
    """
    Visualiza las rutas del modelo DRL y determinista en un mapa con folium.
    Las rutas deben contener columnas de latitud y longitud.
    """

    # Leer y normalizar columnas
    df_drl = pd.read_csv(path_drl)
    df_det = pd.read_csv(path_det)

    rename_cols = {
        'latitude': 'lat',
        'longitude': 'lon',
        'Latitude': 'lat',
        'Longitude': 'lon'
    }
    df_drl.rename(columns={k: v for k, v in rename_cols.items() if k in df_drl.columns}, inplace=True)
    df_det.rename(columns={k: v for k, v in rename_cols.items() if k in df_det.columns}, inplace=True)

    # Verificación
    if 'lat' not in df_drl.columns or 'lon' not in df_drl.columns:
        raise ValueError("La ruta DRL no contiene columnas 'lat' y 'lon'")
    if 'lat' not in df_det.columns or 'lon' not in df_det.columns:
        raise ValueError("La ruta determinista no contiene columnas 'lat' y 'lon'")

    # Punto medio para centrar el mapa
    centro_lat = (df_drl['lat'].iloc[0] + df_det['lat'].iloc[0]) / 2
    centro_lon = (df_drl['lon'].iloc[0] + df_det['lon'].iloc[0]) / 2
    mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=8, tiles='CartoDB Positron')

    # Añadir puntos de inicio y fin
    folium.Marker(
        location=[df_drl['lat'].iloc[0], df_drl['lon'].iloc[0]],
        popup="Inicio DRL", icon=folium.Icon(color='blue')
    ).add_to(mapa)
    folium.Marker(
        location=[df_drl['lat'].iloc[-1], df_drl['lon'].iloc[-1]],
        popup="Fin DRL", icon=folium.Icon(color='darkblue')
    ).add_to(mapa)

    folium.Marker(
        location=[df_det['lat'].iloc[0], df_det['lon'].iloc[0]],
        popup="Inicio Determinista", icon=folium.Icon(color='green')
    ).add_to(mapa)
    folium.Marker(
        location=[df_det['lat'].iloc[-1], df_det['lon'].iloc[-1]],
        popup="Fin Determinista", icon=folium.Icon(color='darkgreen')
    ).add_to(mapa)

    # Añadir líneas
    folium.PolyLine(
        locations=list(zip(df_drl['lat'], df_drl['lon'])),
        color='blue', weight=3, tooltip="Ruta DRL"
    ).add_to(mapa)

    folium.PolyLine(
        locations=list(zip(df_det['lat'], df_det['lon'])),
        color='green', weight=3, tooltip="Ruta Determinista"
    ).add_to(mapa)

    return mapa

import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def calcular_tiempo_aproximado(csv_path, tipo="auto"):
    """
    Calcula el tiempo estimado de una ruta en horas, según el tipo de fichero.
    Args:
        csv_path: ruta al CSV de la ruta
        tipo: "drl", "det" o "auto" (detecta por columnas)
    Returns:
        Tiempo estimado en horas (float)
    """
    df = pd.read_csv(csv_path)

    # Detección automática
    if tipo == "auto":
        if 'lon' in df.columns and 'lat' in df.columns:
            tipo = "drl"
        elif 'longitude' in df.columns and 'latitude' in df.columns:
            tipo = "det"
        else:
            raise ValueError("No se puede detectar el tipo de fichero. Usa 'tipo=\"drl\"' o '\"det\"'.")

    # Columnas y velocidad
    if tipo == "drl":
        coords = df[['lat', 'lon']].values
        velocidades = df['speed'].values
    elif tipo == "det":
        coords = df[['latitude', 'longitude']].values
        velocidades = df['boat_speed'].values if 'boat_speed' in df.columns else df['speed'].values
    else:
        raise ValueError("Tipo no reconocido: usa 'drl' o 'det'")

    # Distancia total
    total_dist_km = 0.0
    for i in range(1, len(coords)):
        total_dist_km += haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])

    velocidad_kn = velocidades.mean()
    velocidad_kmh = velocidad_kn * 1.852

    if velocidad_kmh == 0:
        return float('inf')

    tiempo_horas = total_dist_km / velocidad_kmh
    return tiempo_horas
