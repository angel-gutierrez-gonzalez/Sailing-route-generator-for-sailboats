import requests
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from typing import List, Tuple, Union

# Definición de tipos de seamarks para restricciones y boya de peligro
RESTRICTED_TYPES = {
    "restricted_area","prohibited_area","safety_zone",
    "traffic_separation_scheme","fairway","no_anchor_zone",
    "no_entry_zone","foul_ground"
}
DANGER_BUOY_TYPES = {
    "buoy_isolated_danger","buoy_special_purpose","buoy_safe_water"
}


def fetch_seamark_elements(lat_min: float, lon_min: float, lat_max: float, lon_max: float,
                           timeout: int = 60) -> List[dict]:
    """
    Consulta Overpass API para obtener seamarks (nodes, ways, relations) en un bounding box.
    """
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    query = f"""
[out:json][timeout:{timeout}];
(
  node["seamark:type"]({bbox});
  way["seamark:type"]({bbox});
  relation["seamark:type"]({bbox});
);
out body geom;
"""
    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": query})
    resp.raise_for_status()
    return resp.json().get("elements", [])


def _make_geom(el: dict) -> Union[Point, LineString, Polygon, None]:
    """
    Convierte un elemento de Overpass en geometría Shapely.
    """
    if el["type"] == "node":
        return Point(el["lon"], el["lat"])
    coords = [(c["lon"], c["lat"]) for c in el.get("geometry", [])]
    if not coords:
        return None
    # Si es polígono cerrado
    if len(coords) >= 3 and coords[0] == coords[-1]:
        return Polygon(coords)
    # Si es linea
    return LineString(coords)


def classify_seamarks(elements: List[dict]) -> Tuple[List[Union[Polygon, LineString]], List[Point]]:
    """
    Separa geometrías de seamarks en restricciones y boyas de peligro.
    Retorna:
      - restricted_geoms: lista de Polygon/LineString de áreas restringidas
      - danger_buoys: lista de Point para boyas de peligro
    """
    restricted = []
    buoys = []
    for el in elements:
        seamark_type = el.get("tags", {}).get("seamark:type")
        if not seamark_type:
            continue
        geom = _make_geom(el)
        if geom is None:
            continue
        if seamark_type in RESTRICTED_TYPES:
            restricted.append(geom)
        if seamark_type in DANGER_BUOY_TYPES and isinstance(geom, Point):
            buoys.append(geom)
    return restricted, buoys


def mark_blocked_seamark(mesh_df: pd.DataFrame,
                         restricted_geoms: List[Union[Polygon, LineString]],
                         seguridad: float = 0.01) -> pd.Series:
    """
    Marca en mesh_df los puntos bloqueados por seamarks.
    Seguridad es el buffer en grados.
    """
    union_restr = unary_union(restricted_geoms)
    def is_blocked(row):
        pt = Point(row['longitude'], row['latitude'])
        return (union_restr.intersects(pt)
                or union_restr.distance(pt) <= seguridad)
    return mesh_df.apply(is_blocked, axis=1)


def create_buoy_nodes(buoy_geoms: List[Point]) -> pd.DataFrame:
    """
    Genera DataFrame de boyas de peligro como nodos bloqueados.
    """
    rows = []
    for b in buoy_geoms:
        rows.append({
            'latitude': b.y,
            'longitude': b.x,
            'depth_avg': None,
            'time': None,
            'wind_speed_10m': None,
            'wind_direction_10m': None,
            'navigable': False,
            'blocked_seamark': True,
            'is_buoy': True
        })
    return pd.DataFrame(rows)


def combine_mesh_and_buoys(mesh_df: pd.DataFrame, buoy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatena mesh_df y buoy_df en un único DataFrame de nodos.
    """
    mesh_df['is_buoy'] = False
    return pd.concat([mesh_df, buoy_df], ignore_index=True, sort=False)


def compute_navigable_final(nodes_df: pd.DataFrame) -> pd.Series:
    """
    Combina navegabilidad previa y bloqueo por seamark en navegable_final.
    """
    return nodes_df['navigable'] & (~nodes_df['blocked_seamark'])
