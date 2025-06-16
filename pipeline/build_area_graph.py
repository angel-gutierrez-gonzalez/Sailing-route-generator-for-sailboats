# pipeline/build_area_graph.py
"""
Orquestador para construir el grafo de navegación completo para un área definida por un GeoJSON.
"""
import os
import json
import pickle
import pandas as pd
import networkx as nx
from shapely.geometry import shape, Point
from shapely.ops import unary_union

# Configuración y utilidades
from utils.config import BASE_DIR, SAVE_DIR
from utils.geo_utils import load_new_area
from graph.generate_points import generate_points_in_area
from graph.env_data import batch_evaluate_navigability
from graph.wind_data import batch_evaluate_wind
from graph.nautical_constraints import (
    fetch_seamark_elements, classify_seamarks,
    mark_blocked_seamark, create_buoy_nodes,
    combine_mesh_and_buoys, compute_navigable_final
)
from graph.build_graph import build_weighted_graph


def build_graph_for_area(
    area_geojson: str,
    draft: float,
    w_max: float,
    start_date: str,
    end_date: str,
    paso_lon: float,
    max_neighbors: int = 32,
    neighbor_cells: int = 3,
    alpha_time: float = 1.0,
    beta_comfort: float = 0.1,
    output_path: str = None
) -> nx.DiGraph:
    """
    Construye el grafo de navegación para una única área definida por un GeoJSON.

    Parámetros:
      - area_geojson: ruta al GeoJSON del área seleccionada
      - draft: calado máximo del barco
      - w_max: velocidad de viento máxima navegable (kn)
      - start_date, end_date: fechas para datos horarios de viento
      - paso_lon: espaciado longitudinal (en grados) para muestreo de nodos
      - max_neighbors, neighbor_cells, alpha_time, beta_comfort: parámetros del grafo
      - output_path: ruta donde guardar el grafo .gpickle

    Retorna:
      - G: grafo de tipo nx.DiGraph
    """
    # 1) Cargar la geometría del área
    poly = load_new_area(area_geojson)
    
    # 2) Generar malla de puntos dentro del área
    puntos = generate_points_in_area(poly, paso_lon)

    # 3) Evaluar batimetría y navegabilidad por profundidad
    bathy_results = batch_evaluate_navigability(puntos, draft)

    # 4) Evaluar viento horario (T0) sobre nodos navegables
    wind_results = batch_evaluate_wind(bathy_results, {'w_max': w_max}, start_date, end_date)

    # 5) Preparar DataFrame de nodos base
    mesh_df = pd.DataFrame(wind_results)

    # 6) Aplicar restricciones náuticas
    minx, miny, maxx, maxy = poly.bounds
    elements = fetch_seamark_elements(miny, minx, maxy, maxx)
    restricted, buoys = classify_seamarks(elements)
    mesh_df['blocked_seamark'] = mark_blocked_seamark(mesh_df, restricted)
    buoy_df = create_buoy_nodes(buoys)
    nodes_df = combine_mesh_and_buoys(mesh_df, buoy_df)
    nodes_df['navigable_final'] = compute_navigable_final(nodes_df)

    # 7) Construir grafo ponderado
    G = build_weighted_graph(
        nodes_df,
        polar_df=pd.read_csv(os.path.join("..", "data", 'raw', 'malla', 'polar_diagram.csv')),
        union_restr=unary_union(restricted),
        max_neighbors=max_neighbors,
        neighbor_cells=neighbor_cells,
        alpha_time=alpha_time,
        beta_comfort=beta_comfort
    )

    # 8) Guardar el grafo si se indica
    if output_path is None:
        output_path = os.path.join("..", "data", 'raw', 'grafo', 'area_graph.gpickle')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)

    return G
