# graph/augment_graph.py

import math
import pickle
from typing import Tuple, List
import networkx as nx
from shapely.geometry import LineString, shape
from scipy.spatial import cKDTree
from utils.config import BASE_DIR
from graph.calculate_weights import PolarDiagram
from utils.geo_utils import haversine, bearing


def augment_with_start_end(
    G: nx.DiGraph,
    start: Tuple[float, float],
    end: Tuple[float, float],
    union_restr,
    polar: PolarDiagram,
    max_neighbors: int = 32,
    neighbor_radius: float = 0.02,
    alpha_time: float = 1.0,
    beta_comfort: float = 0.1
) -> nx.DiGraph:
    """
    Crea un nuevo grafo a partir de G añadiendo nodos 'START' y 'END',
    conectándolos a los vecinos válidos con 32 direcciones sectorizadas.

    Args:
      G: grafo original de nx.DiGraph con atributos 'latitude','longitude','wind_speed','wind_dir'
      start, end: tuplas (lat, lon)
      union_restr: geometría Shapely de zonas prohibidas
      polar: instancia de PolarDiagram con tu diagrama polar
      max_neighbors: número de sectores direccionales (default 32)
      neighbor_radius: radio en grados para buscar vecinos cerca
      alpha_time, beta_comfort: parámetros para peso

    Returns:
      G_aug: grafo aumentado
    """
    G_aug = G.copy()
    # Añadir START y END
    for label, coords in [('START', start), ('END', end)]:
        G_aug.add_node(label,
                       latitude=coords[0],
                       longitude=coords[1],
                       wind_speed=G.nodes[next(iter(G.nodes))]['wind_speed'],
                       wind_dir=G.nodes[next(iter(G.nodes))]['wind_dir'])

    # Preparar KDTree de nodos existentes
    coords = [(G_aug.nodes[n]['latitude'], G_aug.nodes[n]['longitude'])
              for n in G.nodes]
    tree = cKDTree(coords)
    labels = list(G.nodes)
    sector_width = 360.0 / max_neighbors
    
    def connect_node(label: str, is_end: bool):
        lat_u = G_aug.nodes[label]['latitude']
        lon_u = G_aug.nodes[label]['longitude']
        Dw    = G_aug.nodes[label]['wind_dir']
        idxs  = tree.query_ball_point([lat_u, lon_u], r=neighbor_radius)
        idxs  = [i for i in idxs]  # incluye vecinos, no filtramos aquí

        for k in range(max_neighbors):
            theta = k * sector_width + sector_width/2
            best, best_dist = None, float('inf')
            for i in idxs:
                v = labels[i]
                lat_v = G_aug.nodes[v]['latitude']
                lon_v = G_aug.nodes[v]['longitude']
                brg = bearing(lon_u, lat_u, lon_v, lat_v)
                if abs((brg - theta + 180) % 360 - 180) > sector_width/2:
                    continue
                d = haversine(lon_u, lat_u, lon_v, lat_v)
                if d < best_dist:
                    best_dist, best = d, v
            if best is None:
                continue

            # Filtrado por zonas prohibidas
            seg = LineString([(lon_u, lat_u),
                              (G_aug.nodes[best]['longitude'],
                               G_aug.nodes[best]['latitude'])])
            if union_restr.intersects(seg):
                continue

            # Cálculo de peso (igual que antes)...
            brg_true   = bearing(lon_u, lat_u,
                                 G_aug.nodes[best]['longitude'],
                                 G_aug.nodes[best]['latitude'])
            twa        = abs(brg_true - Dw)
            boat_speed = polar.get_speed(twa)
            if boat_speed <= 0:
                continue
            time_h  = (best_dist / 1852.0) / boat_speed
            comfort = abs(math.cos(math.radians(twa)))
            weight  = alpha_time * time_h + beta_comfort * comfort

            # **Aquí invertimos la dirección si es END**
            if is_end:
                # arista desde el vecino hacia END
                G_aug.add_edge(best, label,
                               distance_m=best_dist,
                               time_h=time_h,
                               comfort=comfort,
                               weight=weight,
                               heading=brg_true)
            else:
                # arista desde START hacia el vecino, o entre nodos regulares
                G_aug.add_edge(label, best,
                               distance_m=best_dist,
                               time_h=time_h,
                               comfort=comfort,
                               weight=weight,
                               heading=brg_true)

    # Finalmente, lanza las dos conexiones
    connect_node('START', is_end=False)
    connect_node('END',   is_end=True)

    return G_aug
