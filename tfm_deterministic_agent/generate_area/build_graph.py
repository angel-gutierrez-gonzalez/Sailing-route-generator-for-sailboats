import math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import unary_union

class PolarDiagram:
    def __init__(self, polar_df: pd.DataFrame, twa_col='TWA', speed_col=None):
        df = polar_df.copy()
        # Renombrar columnas
        if twa_col not in df.columns:
            raise KeyError(f"TWA column '{twa_col}' not found")
        if speed_col is None:
            candidates = [c for c in df.columns if 'speed' in c.lower()]
            if not candidates:
                raise KeyError("No speed column found")
            speed_col = candidates[0]
        df = df.rename(columns={twa_col: 'TWA', speed_col: 'Speed'})
        # Quitar duplicados y ordenar
        df = df[['TWA','Speed']].drop_duplicates('TWA').sort_values('TWA').reset_index(drop=True)
        self.twas   = df['TWA'].values
        self.speeds = df['Speed'].values

    def get_speed(self, twa: float) -> float:
        """
        Devuelve gamma(w, beta) interpolado en el diagrama polar.
        Extrapola en los extremos sin IndexError.
        """
        # Normalizar twa a [-180,180]
        twa = abs(((twa + 180) % 360) - 180)
        # Interpolar linealmente, extrapolando a los extremos
        return float(np.interp(twa, self.twas, self.speeds))


def haversine(lon1, lat1, lon2, lat2) -> float:
    """
    Devuelve la distancia en MILLAS NÁUTICAS entre dos puntos (φ, λ),
    equivalente a la ecuación (7) del PDF.
    """
    # Radio terrestre medio en metros
    R = 6371000  
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    dist_m = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return dist_m / 1852.0  # convertir metros a millas náuticas


def bearing(lon1, lat1, lon2, lat2) -> float:
    """
    Rumbo verdadero de P->Q en grados [0,360).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    lam1, lam2 = math.radians(lon1), math.radians(lon2)
    y = math.sin(lam2 - lam1) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(lam2 - lam1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def build_weighted_graph(
    nodes_df: pd.DataFrame,
    polar_df: pd.DataFrame,
    union_restr,               # MultiPolygon de zonas NO navegables
    max_neighbors: int = 32,
    neighbor_cells: int = 3,
    alpha_time: float = 1.0,
    beta_comfort: float = 0.1,
    beta_turn: float = 0.0   # <-- nuevo parámetro para almacenar penalización de virada
) -> nx.DiGraph:
    """
    Construye un grafo dirigido ponderado:
      - Tiempo (ec.9)
      - Comodidad (ec.10)
      - (Opcional) beta_turn penalización de cambio de rumbo
    Cada arista guarda:
      distance_nm, time_h, comfort, heading, weight_base, beta_turn.
    """
    polar = PolarDiagram(polar_df)
    G = nx.DiGraph()

    # Filtrar nodos navegables y resetear índice para node_id coherente
    nav = nodes_df[nodes_df['navigable_final']].reset_index(drop=True).copy()
    if len(nav) < 2:
        return G  # Sin suficientes nodos, grafo vacío
    nav['node_id'] = nav.index

    # Añadir nodos al grafo
    for _, r in nav.iterrows():
        G.add_node(int(r.node_id),
                   latitude=r.latitude,
                   longitude=r.longitude,
                   wind_speed=r.wind_speed_10m,
                   wind_dir=r.wind_direction_10m)

    # Preparar KDTree en (lat,lon)
    coords = nav[['latitude','longitude']].values
    tree = cKDTree(coords)

    # Calcular espaciado de la rejilla
    lats = sorted(nav['latitude'].unique())
    lons = sorted(nav['longitude'].unique())
    if len(lats) < 2 or len(lons) < 2:
        return G
    dlat = min(abs(b - a) for a, b in zip(lats, lats[1:]))
    dlon = min(abs(b - a) for a, b in zip(lons, lons[1:]))

    radius_deg = math.sqrt((neighbor_cells*dlat)**2 + (neighbor_cells*dlon)**2)
    sector_width = 360.0 / max_neighbors

    # Crear aristas
    for u in G.nodes:
        lon_u = G.nodes[u]['longitude']
        lat_u = G.nodes[u]['latitude']
        Dw    = G.nodes[u]['wind_dir']

        # vecinos locales
        idxs = tree.query_ball_point([lat_u, lon_u], r=radius_deg)
        idxs = [i for i in idxs if i != u]
        if not idxs:
            continue

        for k in range(max_neighbors):
            theta = k * sector_width + sector_width/2
            best_v, best_dist_nm = None, float('inf')

            for i in idxs:
                v = int(nav.at[i, 'node_id'])
                lon_v = G.nodes[v]['longitude']
                lat_v = G.nodes[v]['latitude']
                brg = bearing(lon_u, lat_u, lon_v, lat_v)
                diff = abs((brg - theta + 180) % 360 - 180)
                if diff > sector_width/2:
                    continue

                d_nm = haversine(lon_u, lat_u, lon_v, lat_v)
                if d_nm < best_dist_nm:
                    best_dist_nm, best_v = d_nm, v

            if best_v is None:
                continue

            # comprobar intersección con zonas no navegables
            seg = LineString([
                (lon_u, lat_u),
                (G.nodes[best_v]['longitude'], G.nodes[best_v]['latitude'])
            ])
            if union_restr.intersects(seg):
                continue

            # calcular costes
            brg_true = bearing(lon_u, lat_u,
                               G.nodes[best_v]['longitude'],
                               G.nodes[best_v]['latitude'])
            twa = abs(brg_true - Dw)
            boat_speed = polar.get_speed(twa)
            if boat_speed <= 0:
                continue

            time_h    = best_dist_nm / boat_speed
            comfort   = abs(math.cos(math.radians(twa)))
            # aquí NO conocemos el heading previo: beta_turn se aplicará en la búsqueda
            weight_base = alpha_time * time_h + beta_comfort * comfort

            G.add_edge(u, best_v,
                       distance_nm=best_dist_nm,
                       time_h=time_h,
                       comfort=comfort,
                       heading=brg_true,
                       weight_base=weight_base,
                       beta_turn=beta_turn)

    return G
