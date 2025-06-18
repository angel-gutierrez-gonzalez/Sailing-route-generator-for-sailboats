import math

# Parámetros del elipsoide de Bessel
a_metros = 6377397.155
# Semieje mayor en minutos de arco (1′ = 1 Milla náutica = 1852 m)
_a = a_metros / 1852.0
# Excentricidad del elipsoide de Bessel
e = 0.081696831215

def generate_mesh_mercator(lat_min: float, lat_max: float,
                            lon_min: float, lon_max: float,
                            paso_lon: float, N_lat: int) -> list:
    """
    Genera una malla de nodos P_k = (φ_i, λ_j) usando:
    - proyección de Mercator expandida para latitud (ecuación (3) del artículo).
    - Δλ uniforme en longitud (ecuación (4) del artículo).

    Parámetros:
    - lat_min, lat_max, lon_min, lon_max en grados.
    - paso_lon: espaciado en longitud (°).
    - N_lat: número de intervalos en latitud.

    Devuelve:
    - Lista de tuplas (lat, lon) con los puntos de la malla.
    """
    # 1) Calcular N_lon y ajustar lon_max
    N_lon = int(math.floor((lon_max - lon_min) / paso_lon))
    lon_max_adj = lon_min + N_lon * paso_lon

    # 2) Convertir extremos de latitud a radianes
    phi1 = math.radians(lat_min)
    phi2 = math.radians(lat_max)

    # Función de latitud expandida V(φ)
    def mercator_expanded(phi: float) -> float:
        term1 = math.tan(math.pi/4 + phi/2)
        term2 = ((1 - e * math.sin(phi)) / (1 + e * math.sin(phi))) ** (e/2)
        return _a * math.log(term1 * term2)

    # Inversa V -> φ por Newton–Raphson
    def invert_mercator(V: float) -> float:
        # Estimación inicial (esférica)
        phi = 2 * math.atan(math.exp(V / _a)) - math.pi/2
        for _ in range(3):
            sinphi = math.sin(phi)
            W = math.tan(math.pi/4 + phi/2) * ((1 - e * sinphi)/(1 + e * sinphi))**(e/2)
            f = _a * math.log(W) - V
            # derivada por diferencia finita
            h = 1e-8
            sinphi_h = math.sin(phi + h)
            W_h = math.tan(math.pi/4 + (phi+h)/2) * ((1 - e * sinphi_h)/(1 + e * sinphi_h))**(e/2)
            df = ((_a * math.log(W_h) - V) - f) / h
            phi -= f/df
        return phi

    # 3) Calcular V1, Vm y vector V
    V1 = mercator_expanded(phi1)
    Vm = mercator_expanded(phi2)
    delta_V = (Vm - V1) / N_lat
    V_arr = [V1 + i * delta_V for i in range(N_lat + 1)]

    # 4) Invertir a latitudes (en grados)
    lats = [math.degrees(invert_mercator(V)) for V in V_arr]

    # 5) Calcular longitudes uniformes
    delta_lon = (lon_max_adj - lon_min) / N_lon
    lons = [lon_min + j * delta_lon for j in range(N_lon + 1)]

    # 6) Generar lista de nodos (lat, lon)
    return [(lat, lon) for lat in lats for lon in lons]
