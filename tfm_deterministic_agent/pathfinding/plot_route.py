# visualization/plot_route.py

import folium
from typing import List, Any
import networkx as nx


def plot_route_on_graph(
    G: nx.Graph,
    path: List[Any],
    route_color: str = 'red',
    node_color: str = 'blue',
    start_color: str = 'green',
    end_color: str = 'red',
    zoom_start: int = 10
) -> folium.Map:
    """
    Dibuja la ruta calculada sobre un mapa interactivo usando las coordenadas de los nodos en el grafo.

    Args:
        G: Grafo con atributos 'latitude' y 'longitude' en cada nodo.
        path: Lista de nodos que conforman la ruta (labels o IDs de nodos en G).
        route_color: Color de la línea de ruta.
        node_color: Color de los nodos intermedios.
        start_color: Color para el nodo de inicio.
        end_color: Color para el nodo de fin.
        zoom_start: Nivel inicial de zoom.

    Returns:
        Folium.Map con la ruta y nodos dibujados.
    """
    if not path:
        raise ValueError("Ruta vacía")

    # Extraer coordenadas
    coords = []
    for n in path:
        if n not in G.nodes:
            raise KeyError(f"Nodo {n} no encontrado en el grafo")
        lat = G.nodes[n].get('latitude')
        lon = G.nodes[n].get('longitude')
        coords.append((lat, lon))

    # Mapa centrado en primer punto
    m = folium.Map(location=coords[0], zoom_start=zoom_start)

    # Dibujar línea de ruta
    folium.PolyLine(
        locations=coords,
        color=route_color,
        weight=3,
        opacity=0.8,
        tooltip="Ruta óptima"
    ).add_to(m)

    # Dibujar nodos intermedios
    for lat, lon in coords:
        folium.CircleMarker(
            location=(lat, lon),
            radius=3,
            color=node_color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    # Marcar inicio y fin
    folium.CircleMarker(
        location=coords[0],
        radius=6,
        color=start_color,
        fill=True,
        fill_color=start_color,
        tooltip="Inicio"
    ).add_to(m)
    folium.CircleMarker(
        location=coords[-1],
        radius=6,
        color=end_color,
        fill=True,
        fill_color=end_color,
        tooltip="Destino"
    ).add_to(m)

    return m
