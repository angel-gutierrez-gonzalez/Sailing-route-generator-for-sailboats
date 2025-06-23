import numpy as np

def compute_reward(position, goal, boat_speed, wind_dir, heading, config=None, debug=False):
    """
    Calcula la recompensa en función de: distancia al destino, escora esperada y eficiencia del rumbo.

    Args:
        position: [lat, lon] actual
        goal: [lat, lon] destino
        boat_speed: velocidad actual del barco (nudos)
        wind_dir: dirección del viento (°)
        heading: rumbo del barco (°)
        config: diccionario opcional con pesos
        debug: bool para imprimir logs

    Returns:
        reward (float)
    """
    cfg = config or {
        'w_distance': 1.0,
        'w_heel': 0.2,
        'w_efficiency': 1.5  # mayor peso para alineación con el destino
    }

    # 1. Distancia al destino (Haversine simple en grados)
    dist = np.linalg.norm(np.array(position) - np.array(goal))
    reward_dist = -cfg['w_distance'] * dist

    # 2. Penalización por escora (heel) basada en ángulo de viento relativo
    rel_wind = (wind_dir - heading) % 360
    heel_angle = estimate_heel(rel_wind, boat_speed)  # aproximación
    penalized_heel = max(0, heel_angle - 15)  # solo penaliza si pasa de 15°
    reward_heel = -cfg['w_heel'] * penalized_heel

    # 3. Eficiencia: cuanto más alineado está el rumbo con la línea hacia el destino
    goal_angle = np.degrees(np.arctan2(goal[1] - position[1], goal[0] - position[0])) % 360
    angle_diff = min(abs(heading - goal_angle), 360 - abs(heading - goal_angle))
    reward_eff = -cfg['w_efficiency'] * angle_diff / 180  # normalizado

    # 4. Bonificación si se acerca mucho al destino
    reward_bonus = 50 if dist < 0.5 else 0

    if debug:
        print(f"[STEP LOG] dist={dist:.4f}, heel={heel_angle:.2f}°, angle_diff={angle_diff:.1f}°, speed={boat_speed:.2f} kn")

    return reward_dist + reward_heel + reward_eff + reward_bonus


def estimate_heel(relative_wind_angle, boat_speed):
    """
    Estima la escora del barco como función del ángulo de viento y velocidad.
    """
    if relative_wind_angle > 180:
        relative_wind_angle = 360 - relative_wind_angle

    factor = np.sin(np.radians(relative_wind_angle))
    heel = factor * boat_speed / 2  # simplificación, típicamente ~0-25°
    return np.clip(heel, 0, 25)
