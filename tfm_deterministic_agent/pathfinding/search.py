# pathfinding/search.py

import pickle
import types
import sys
import numpy as np
import networkx as nx
import os
import time
import heapq
from typing import Any, Dict

def load_gpickle_compatible(path: str) -> nx.Graph:
    """
    Carga un archivo .gpickle compatible con versiones antiguas de NetworkX y NumPy.
    """
    if 'numpy._core' not in sys.modules:
        stub = types.ModuleType('numpy._core')
        stub.__dict__.update(np.__dict__)
        sys.modules['numpy._core'] = stub
        for sub in ('multiarray', 'numeric', 'numerictypes'):
            sys.modules[f'numpy._core.{sub}'] = getattr(np.core, sub, stub)

    with open(path, 'rb') as f:
        return pickle.load(f)

def run_dijkstra(
    G: nx.Graph,
    source: Any,
    target: Any,
    weight: str = "weight"
) -> Dict[str, Any]:
    """
    Ejecuta el algoritmo de Dijkstra y devuelve un dict con:
      - path: lista de nodos desde source a target
      - cost: coste total (según atributo 'weight')
      - hops: número de saltos (aristas)
      - time: tiempo de cómputo en segundos
    """
    t0 = time.time()
    path = nx.dijkstra_path(G, source=source, target=target, weight=weight)
    cost = nx.dijkstra_path_length(G, source=source, target=target, weight=weight)
    t1 = time.time()

    return {
        "path": path,
        "cost": cost,
        "hops": len(path) - 1,
        "time": t1 - t0
    }

def shortest_path_with_turn_penalty(
    G: nx.Graph,
    source: Any,
    target: Any,
    beta_turn: float = 0.0
) -> Dict[str, Any]:
    """
    Encuentra la ruta de coste mínimo entre source y target
    en Grafo dirigido G, incluyendo penalización de virada.

    Cada arista (u→v) de G debe tener:
      - 'weight_base': tiempo+confort
      - 'heading': rumbo verdadero de u→v en grados
    """
    # Cola de prioridad: (coste_acum, nodo, heading_prev, path)
    pq = [(0.0, source, None, [source])]
    best = {}  # mejor coste visto para (nodo, heading_prev)

    while pq:
        cost_u, u, hd_prev, path = heapq.heappop(pq)
        if u == target:
            return {"path": path, "cost": cost_u, "hops": len(path)-1}

        key = (u, hd_prev)
        if key in best and best[key] <= cost_u:
            continue
        best[key] = cost_u

        for v, attrs in G[u].items():
            wb    = attrs.get('weight_base', attrs.get('weight', 0.0))
            hd_uv = attrs.get('heading', 0.0)
            turn  = 0.0 if hd_prev is None else beta_turn * abs(hd_prev - hd_uv)
            new_cost = cost_u + wb + turn
            heapq.heappush(pq, (new_cost, v, hd_uv, path + [v]))

    # Si no hay camino:
    return {"path": None, "cost": float('inf'), "hops": 0}
