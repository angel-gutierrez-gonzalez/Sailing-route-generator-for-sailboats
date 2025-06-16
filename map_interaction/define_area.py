from shapely.geometry import Polygon, Point

def define_study_area(start, end, buffer_distance=0.01):
    # Extraer coordenadas
    start_point = Point(start[1], start[0])  # (lon, lat)
    end_point = Point(end[1], end[0])

    # Calcular límites del área de estudio
    minx = min(start_point.x, end_point.x) - buffer_distance
    miny = min(start_point.y, end_point.y) - buffer_distance
    maxx = max(start_point.x, end_point.x) + buffer_distance
    maxy = max(start_point.y, end_point.y) + buffer_distance

    # Crear el polígono rectangular
    bounding_box = Polygon([
        (minx, miny), 
        (minx, maxy), 
        (maxx, maxy), 
        (maxx, miny), 
        (minx, miny)
    ])

    return bounding_box