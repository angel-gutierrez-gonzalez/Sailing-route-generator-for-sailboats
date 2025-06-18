# map_interaction/select_points.py

import os
import json
import folium
from folium.plugins import Draw
from shapely.geometry import shape, Point
from utils.config import SAVE_DIR


def create_map(area_geojson_path=os.path.join(SAVE_DIR, "selection.geojson")):
    """
    Genera un mapa centrado en el área de estudio, permitiendo seleccionar dos puntos (origen y destino).
    """
    # Cargar área de estudio
    if not os.path.isfile(area_geojson_path):
        raise FileNotFoundError(f"No se encontró el área de estudio en: {area_geojson_path}")
    
    with open(area_geojson_path, 'r', encoding='utf-8') as f:
        area_geojson = json.load(f)

    area_shape = shape(area_geojson['features'][0]['geometry'])
    bounds = area_shape.bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')

    # Capas base
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

    # Área de estudio como capa
    folium.GeoJson(area_geojson, name='Área de estudio', style_function=lambda x: {
        'color': 'blue', 'fillColor': 'lightblue', 'fillOpacity': 0.3
    }).add_to(m)

    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Plugin Draw (solo puntos)
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


def validate_points_in_area(points, area_geojson_path):
    """
    Verifica si todos los puntos dados (lat, lon) están dentro del área.
    """
    with open(area_geojson_path, 'r', encoding='utf-8') as f:
        area_geojson = json.load(f)

    area_shape = shape(area_geojson['features'][0]['geometry'])

    for lat, lon in points:
        if not area_shape.contains(Point(lon, lat)):
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


def save_geojson_to_disk(geojson: dict, filename: str = 'selected_points.geojson'):
    """
    Guarda un GeoJSON en disco dentro del SAVE_DIR.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, filename)
    if 'features' not in geojson:
        geojson = {'type': 'FeatureCollection', 'features': [geojson]}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)


def load_selection_geojson(filename: str = 'selected_points.geojson') -> dict:
    """
    Carga un archivo GeoJSON desde SAVE_DIR.
    """
    path = os.path.join(SAVE_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el fichero: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_validated_start_end_points(
    points_file: str = "selected_points.geojson",
    area_file: str = "selection.geojson"
):
    """
    Carga los puntos seleccionados y valida que estén dentro del área de estudio.
    Devuelve tuplas (lat, lon) de inicio y fin.
    """
    # Cargar puntos
    points_path = os.path.join(SAVE_DIR, points_file)
    if not os.path.exists(points_path):
        raise FileNotFoundError(f"No se encontró el archivo de puntos: {points_path}")
    with open(points_path, "r", encoding="utf-8") as f:
        points_geojson = json.load(f)

    features = points_geojson.get("features", [])
    if len(features) < 2:
        raise ValueError("Se requieren exactamente dos puntos seleccionados (inicio y fin).")

    lon1, lat1 = features[0]["geometry"]["coordinates"]
    lon2, lat2 = features[1]["geometry"]["coordinates"]
    start = (lat1, lon1)
    end = (lat2, lon2)

    # Validar dentro del área
    if not validate_points_in_area([start, end], os.path.join(SAVE_DIR, area_file)):
        raise ValueError("Alguno de los puntos está fuera del área de estudio.")

    return start, end
