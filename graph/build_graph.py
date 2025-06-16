# graph/build_graph.py
"""
Construcción de un grafo ponderado con direcciones sectorizadas según el método de 32 movimientos.
"""
import math
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from utils.geo_utils import haversine, bearing
from graph.calculate_weights import PolarDiagram


def build_weighted_graph(
    nodes_df,
    polar_df,
    union_restr,
    max_neighbors: int = 32,
    neighbor_cells: int = 3,
    alpha_time: float = 1.0,
    beta_comfort: float = 0.1
) -> nx.DiGraph:
    """
    Construye un grafo dirigido ponderado para navegación:

    Parámetros:
      - nodes_df: pd.DataFrame con columnas ['latitude', 'longitude',
        'wind_speed_10m', 'wind_direction_10m', 'navigable_final']
      - polar_df: pd.DataFrame con tu diagrama polar (TWA vs Speed).
      - union_restr: geometría Shapely de zonas NO navegables.
      - max_neighbors: número de sectores direccionales (32).
      - neighbor_cells: radio en celdas para buscar vecinos.
      - alpha_time: coeficiente para tiempo en el peso.
      - beta_comfort: coeficiente para comodidad en el peso.

    Retorna:
      - G: networkx.DiGraph con nodos y aristas.
    """
    polar = PolarDiagram(polar_df)
    G = nx.DiGraph()

    # 1) Filtrar nodos navegables y asignar ID
    nav = nodes_df[nodes_df['navigable_final']].reset_index(drop=True).copy()
    nav['node_id'] = nav.index

    # 2) Añadir nodos con atributos
    for _, r in nav.iterrows():
        G.add_node(
            int(r.node_id),
            latitude=r.latitude,
            longitude=r.longitude,
            wind_speed=r.wind_speed_10m,
            wind_dir=r.wind_direction_10m
        )

    # 3) Preparar KDTree para vecinos locales
    coords = nav[['latitude', 'longitude']].values
    tree = cKDTree(coords)

    # 4) Calcular espaciamiento y radio de búsqueda
    lats = sorted(nav['latitude'].unique())
    lons = sorted(nav['longitude'].unique())
    dlat = min(abs(b - a) for a, b in zip(lats, lats[1:]))
    dlon = min(abs(b - a) for a, b in zip(lons, lons[1:]))
    radius_deg = math.sqrt((neighbor_cells * dlat) ** 2 + (neighbor_cells * dlon) ** 2)

    # 5) Definir ancho angular de cada sector
    sector_width = 360.0 / max_neighbors

    # 6) Para cada nodo, buscar vecinos y sectorizar
    labels = list(G.nodes)
    for u in labels:
        lon_u = G.nodes[u]['longitude']
        lat_u = G.nodes[u]['latitude']
        Dw    = G.nodes[u]['wind_dir']

        # Índices de vecinos en el radio
        idxs = tree.query_ball_point([lat_u, lon_u], r=radius_deg)
        idxs = [i for i in idxs if i != u]
        if not idxs:
            continue

        # Por cada sector, elegir el vecino más cercano
        for k in range(max_neighbors):
            theta = k * sector_width + sector_width / 2
            best_v, best_dist = None, float('inf')

            for i in idxs:
                v = int(nav.at[i, 'node_id'])
                lon_v = G.nodes[v]['longitude']
                lat_v = G.nodes[v]['latitude']

                # Rumbo u->v y filtro sectorial
                brg = bearing(lon_u, lat_u, lon_v, lat_v)
                diff = abs((brg - theta + 180) % 360 - 180)
                if diff > sector_width / 2:
                    continue

                # Distancia real
                d = haversine(lon_u, lat_u, lon_v, lat_v)
                if d < best_dist:
                    best_dist, best_v = d, v

            if best_v is None:
                continue

            # Filtrar aristas que cruzan zona prohibida
            seg = LineString([
                (lon_u, lat_u),
                (G.nodes[best_v]['longitude'], G.nodes[best_v]['latitude'])
            ])
            if union_restr.intersects(seg):
                continue

            # Calcular atributos de arista
            brg_true = bearing(
                lon_u, lat_u,
                G.nodes[best_v]['longitude'],
                G.nodes[best_v]['latitude']
            )
            twa = abs(brg_true - Dw)
            boat_speed = polar.get_speed(twa)
            if boat_speed <= 0:
                continue

            time_h = (best_dist / 1852.0) / boat_speed
            comfort = abs(math.cos(math.radians(twa)))
            weight = alpha_time * time_h + beta_comfort * comfort

            G.add_edge(
                u, best_v,
                distance_m=best_dist,
                time_h=time_h,
                comfort=comfort,
                weight=weight,
                heading=brg_true
            )

    return G
