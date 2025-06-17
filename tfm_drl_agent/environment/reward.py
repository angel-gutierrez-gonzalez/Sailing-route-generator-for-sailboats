import numpy as np

def compute_reward(position, goal, boat_speed, wind_dir, heading, config=None):
    """
    Calcula la recompensa en función de: distancia al destino, escora esperada y eficiencia del rumbo.

    Args:
        position: [lat, lon] actual
        goal: [lat, lon] destino
        boat_speed: velocidad actual del barco (nudos)
        wind_dir: dirección del viento (°)
        heading: rumbo del barco (°)
        config: diccionario opcional con pesos

    Returns:
        reward (float)
    """
    cfg = config or {
        'w_distance': 1.0,
        'w_heel': 0.2,
        'w_efficiency': 0.5
    }

    # 1. Distancia al destino (Haversine simple en grados)
    dist = np.linalg.norm(np.array(position) - np.array(goal))
    reward_dist = -cfg['w_distance'] * dist

    # 2. Penalización por escora (heel) basada en ángulo de viento relativo
    rel_wind = (wind_dir - heading) % 360
    heel_angle = estimate_heel(rel_wind, boat_speed)  # aproximación
    reward_heel = -cfg['w_heel'] * heel_angle

    # 3. Eficiencia: cuanto más alineado está el rumbo con la línea hacia el destino
    goal_angle = np.degrees(np.arctan2(goal[1] - position[1], goal[0] - position[0])) % 360
    angle_diff = min(abs(heading - goal_angle), 360 - abs(heading - goal_angle))
    reward_eff = -cfg['w_efficiency'] * angle_diff / 180  # normalizado

    return reward_dist + reward_heel + reward_eff


def estimate_heel(relative_wind_angle, boat_speed):
    """
    Estima la escora del barco como función del ángulo de viento y velocidad.
    """
    if relative_wind_angle > 180:
        relative_wind_angle = 360 - relative_wind_angle

    factor = np.sin(np.radians(relative_wind_angle))
    heel = factor * boat_speed / 2  # simplificación, típicamente ~0-20°
    return np.clip(heel, 0, 25)
