import os
import json

def get_area_bounds(path_geojson: str) -> tuple:
    """
    Lee un GeoJSON de selección rectangular y devuelve:
    lat_min, lat_max, lon_min, lon_max
    """
    with open(path_geojson, 'r', encoding='utf-8') as f:
        data = json.load(f)
    coords = data['features'][0]['geometry']['coordinates'][0]
    lon1, lat1 = coords[0]
    lon2, lat2 = coords[2]
    lat_min = min(lat1, lat2)
    lat_max = max(lat1, lat2)
    lon_min = min(lon1, lon2)
    lon_max = max(lon1, lon2)
    return lat_min, lat_max, lon_min, lon_max