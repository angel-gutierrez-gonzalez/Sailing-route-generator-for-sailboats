# pathfinding/search.py

import pickle
import types
import sys
import numpy as np
import networkx as nx
import os
import time
from typing import Any, Dict
# from utils.config import BASE_DIR   <-- no se usa, puedes eliminarlo


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
    # Ruta y coste
    path = nx.dijkstra_path(G, source=source, target=target, weight=weight)
    cost = nx.dijkstra_path_length(G, source=source, target=target, weight=weight)
    t1 = time.time()

    return {
        "path": path,
        "cost": cost,
        "hops": len(path) - 1,
        "time": t1 - t0
    }
