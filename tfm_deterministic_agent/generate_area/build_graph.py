import math
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import unary_union
from tqdm import tqdm
import pandas as pd

class PolarDiagram:
    """
    Interpola velocidad de barco γ(wind_speed, twa) a partir
    de un CSV semicolon cuya primera columna es TWA y las
    siguientes columnas son distintos TWS.
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, sep=';')
        first = df.columns[0]
        df = df.rename(columns={first: 'TWA'})
        self.twa = df['TWA'].values.astype(float)
        self.tws = np.array([float(c) for c in df.columns[1:]])
        self.matrix = df.iloc[:,1:].values.astype(float)

    def get_speed(self, twa: float, tws: float) -> float:
        twa_rel = abs(((twa + 180) % 360) - 180)
        twa_rel = np.clip(twa_rel, self.twa[0], self.twa[-1])
        tws_clamped = np.clip(tws, self.tws[0], self.tws[-1])

        # Interpola en TWA (filas)
        i = np.searchsorted(self.twa, twa_rel)
        if i == 0:
            row = self.matrix[0]
        elif i >= len(self.twa):
            row = self.matrix[-1]
        else:
            t0, t1 = self.twa[i-1], self.twa[i]
            w = (twa_rel - t0) / (t1 - t0)
            row = (1-w)*self.matrix[i-1] + w*self.matrix[i]

        # Interpola en TWS (columnas)
        j = np.searchsorted(self.tws, tws_clamped)
        if j == 0:
            return float(row[0])
        elif j >= len(self.tws):
            return float(row[-1])
        else:
            s0, s1 = self.tws[j-1], self.tws[j]
            w2 = (tws_clamped - s0) / (s1 - s0)
            return float((1-w2)*row[j-1] + w2*row[j])

def haversine(lon1, lat1, lon2, lat2) -> float:
    """
    Distancia en millas náuticas entre (lat1,lon1) y (lat2,lon2).
    """
    R = 6371000
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    dist_m = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return dist_m / 1852.0

def bearing(lon1, lat1, lon2, lat2) -> float:
    phi1, phi2 = map(math.radians, (lat1, lat2))
    lam1, lam2 = map(math.radians, (lon1, lon2))
    y = math.sin(lam2 - lam1) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(lam2 - lam1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def build_weighted_graph(
    nodes_df,
    polar_input,            # Ahora puede ser str (ruta CSV) o PolarDiagram
    union_restr,            # MultiPolígono prohibido
    max_neighbors: int = 32,
    neighbor_cells: int = 3,
    alpha_time: float = 1.0,
    beta_comfort: float = 0.1,
    beta_turn: float = 0.0
) -> nx.DiGraph:
    """
    Construye un DiGraph con:
      - weight_base = alpha_time*time + beta_comfort*comfort
      - heading y beta_turn en cada arista
    Utiliza tqdm para mostrar progreso.
    """
    # 1) Instanciar polar
    if isinstance(polar_input, PolarDiagram):
        polar = polar_input
    else:
        polar = PolarDiagram(polar_input)

    # 2) Filtrar nodos navegables
    nav = nodes_df[nodes_df['navigable_final']].reset_index(drop=True).copy()
    if len(nav) < 2:
        return nx.DiGraph()
    nav['node_id'] = nav.index

    # 3) Crear grafo y añadir nodos
    G = nx.DiGraph()
    for _, r in nav.iterrows():
        G.add_node(int(r.node_id),
                   latitude=r.latitude,
                   longitude=r.longitude,
                   wind_speed=r.wind_speed_10m,
                   wind_dir=r.wind_direction_10m)

    # 4) KDTree para vecinos
    coords = nav[['latitude','longitude']].values
    tree = cKDTree(coords)

    # 5) Cálculo espaciado de malla
    lats = sorted(nav['latitude'].unique())
    lons = sorted(nav['longitude'].unique())
    if len(lats)<2 or len(lons)<2:
        return G
    dlat = min(b - a for a,b in zip(lats, lats[1:]))
    dlon = min(b - a for a,b in zip(lons, lons[1:]))
    radius = math.hypot(neighbor_cells*dlat, neighbor_cells*dlon)
    sector = 360.0 / max_neighbors

    # 6) Construcción de aristas con barra de progreso
    for u in tqdm(nav['node_id'], desc="Construyendo grafo", unit="nodo"):
        lon_u = G.nodes[u]['longitude']
        lat_u = G.nodes[u]['latitude']
        Dw    = G.nodes[u]['wind_dir']
        tws   = G.nodes[u]['wind_speed']  # velocidad del viento en nudos

        idxs = tree.query_ball_point([lat_u, lon_u], r=radius)
        idxs = [i for i in idxs if i!=u]
        if not idxs:
            continue

        for k in range(max_neighbors):
            theta = k*sector + sector/2
            best_v, best_dnm = None, float('inf')
            for i in idxs:
                v = int(nav.at[i,'node_id'])
                lon_v, lat_v = G.nodes[v]['longitude'], G.nodes[v]['latitude']
                brg = bearing(lon_u, lat_u, lon_v, lat_v)
                if abs((brg-theta+180)%360 - 180) > sector/2:
                    continue
                dnm = haversine(lon_u, lat_u, lon_v, lat_v)
                if dnm < best_dnm:
                    best_dnm, best_v = dnm, v
            if best_v is None:
                continue

            seg = LineString([(lon_u, lat_u),
                              (G.nodes[best_v]['longitude'],
                               G.nodes[best_v]['latitude'])])
            if union_restr.intersects(seg):
                continue

            # 7) Costes
            brg_true = bearing(lon_u, lat_u,
                               G.nodes[best_v]['longitude'],
                               G.nodes[best_v]['latitude'])
            twa       = abs(brg_true - Dw)
            boat_spd  = polar.get_speed(twa, tws)
            if boat_spd <= 0:
                continue

            time_h     = best_dnm / boat_spd
            comfort    = abs(math.cos(math.radians(twa)))
            w_base     = alpha_time*time_h + beta_comfort*comfort

            G.add_edge(u, best_v,
                       distance_nm=best_dnm,
                       time_h=time_h,
                       comfort=comfort,
                       heading=brg_true,
                       weight_base=w_base,
                       beta_turn=beta_turn)

    return G
