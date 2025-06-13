# Dentro de map_interaction/select_points.py

import os
import json
import folium
from folium.plugins import Draw
from shapely.geometry import shape, Point
from utils.config import SAVE_DIR


import os
import json
import folium
from folium.plugins import Draw
from shapely.geometry import shape
from utils.config import SAVE_DIR


def create_map(area_geojson_path=os.path.join(SAVE_DIR, "selection.geojson")):
    """
    Genera un mapa centrado y ajustado automáticamente al área de estudio definida en el GeoJSON.
    Permite seleccionar dos puntos con el plugin Draw.
    """
    # Cargar el área del geojson
    if not os.path.isfile(area_geojson_path):
        raise FileNotFoundError(f"No se encontró el área de estudio en: {area_geojson_path}")
    
    with open(area_geojson_path, 'r', encoding='utf-8') as f:
        area_geojson = json.load(f)

    area_shape = shape(area_geojson['features'][0]['geometry'])
    bounds = area_shape.bounds  # (minx, miny, maxx, maxy)

    # Calcular centro del área para inicializar el mapa
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')

    # Capa de Seamarks y EMODnet
    folium.TileLayer(
        tiles='https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png',
        attr='© OpenSeaMap contributors',
        name='Seamarks',
        overlay=True,
        control=True
    ).add_to(m)

    folium.raster_layers.WmsTileLayer(
        url='https://ows.emodnet-bathymetry.eu/wms?',
        layers='emodnet:contours',
        fmt='image/png',
        transparent=True,
        version='1.3.0',
        attr='© EMODnet Bathymetry',
        name='Contornos (EMODnet)',
        overlay=True,
        control=True,
        opacity=1.0
    ).add_to(m)

    # Capa del área de estudio
    folium.GeoJson(area_geojson, name='Área de estudio', style_function=lambda x: {
        'color': 'blue', 'fillColor': 'lightblue', 'fillOpacity': 0.3
    }).add_to(m)

    # Ajustar la vista para que encaje con el área
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Dibujo de puntos
    drawn_items = folium.FeatureGroup(name='Drawn Items')
    m.add_child(drawn_items)

    draw = Draw(
        export=True,
        filename='selected_points.geojson',
        feature_group=drawn_items,
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'rectangle': False,
            'circlemarker': False,
            'marker': True,
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)

    folium.LayerControl().add_to(m)

    return m


def validate_points_in_area(points: list[tuple[float, float]], area_geojson_path: str) -> bool:
    """
    Valida si todos los puntos están dentro del polígono del área de estudio.
    """
    with open(area_geojson_path, 'r', encoding='utf-8') as f:
        area_geojson = json.load(f)

    area_geom = shape(area_geojson['features'][0]['geometry'])

    for lat, lon in points:
        p = Point(lon, lat)
        if not area_geom.contains(p):
            return False
    return True


def process_geojson(geojson: dict, area_geojson_path: str = os.path.join(SAVE_DIR, 'selection.geojson')):
    """
    Extrae las coordenadas de dos marcadores de un GeoJSON y valida que estén en el área.
    Devuelve tuplas start y end.
    """
    if 'features' not in geojson and geojson.get('type') == 'Feature':
        geojson = {'type': 'FeatureCollection', 'features': [geojson]}

    features = geojson.get('features', [])
    if len(features) < 2:
        raise ValueError("Se requieren dos marcadores para definir inicio y fin de ruta.")

    lon1, lat1 = features[0]['geometry']['coordinates']
    lon2, lat2 = features[1]['geometry']['coordinates']

    start = (lat1, lon1)
    end = (lat2, lon2)

    if not validate_points_in_area([start, end], area_geojson_path):
        raise ValueError("Ambos puntos deben estar dentro del área de estudio.")

    return start, end


def save_geojson_to_disk(geojson: dict, filename: str = 'selection.geojson'):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, filename)
    if 'features' not in geojson:
        geojson = {'type': 'FeatureCollection', 'features': [geojson]}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)


def load_selection_geojson(filename: str = 'selection.geojson') -> dict:
    path = os.path.join(SAVE_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el fichero de selección en: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
