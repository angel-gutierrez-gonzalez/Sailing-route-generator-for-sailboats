import gym
import numpy as np
from gym import spaces

from environment.reward import compute_reward


class SailingEnv(gym.Env):
    """
    Entorno personalizado de navegación a vela compatible con OpenAI Gym.
    Estado: [lat, lon, heading, speed, wind_dir, wind_speed]
    Acción: [delta_heading, delta_speed] (discreta o continua)
    """
    def __init__(self, config):
        super(SailingEnv, self).__init__()

        self.config = config
        self.wind_field = config['wind']  # instancia de WindField
        self.polar_diagram = config['polar_diagram']  # instancia de PolarDiagram
        self.grid = config.get('grid', None)
        self.goal = np.array(config['goal'], dtype=np.float32)
        self.static_wind = config.get('static_wind', False)
        self.static_wind_values = None

        # Espacio de observación
        low = np.array([0, 0, 0, 0, 0, 0])
        high = np.array([1, 1, 360, 20, 360, 20])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Espacio de acción
        if config['continuous']:
            self.action_space = spaces.Box(
                low=np.array([-10, -1]), high=np.array([10, 1]), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(9)  # 3 heading x 3 speed

        self.reset()

    def reset(self):
        self.position = np.array(self.config['start'], dtype=np.float32)
        self.heading = 90.0
        self.speed = 5.0
        self.t = 0  # tiempo en minutos desde inicio

        wind_dir, wind_speed = self.wind_field.get(self.position, self.t)
        if self.static_wind:
            self.static_wind_values = (wind_dir, wind_speed)

        self.state = np.array([
            *self.position,
            self.heading,
            self.speed,
            wind_dir,
            wind_speed
        ], dtype=np.float32)

        return self.state

    def step(self, action):
        # Acción: cambio en rumbo y velocidad
        if self.config['continuous']:
            delta_heading, delta_speed = action
        else:
            delta_heading, delta_speed = self.decode_discrete_action(action)

        self.heading = (self.heading + delta_heading) % 360
        self.speed = np.clip(self.speed + delta_speed, 0, 20)

        # Obtener condiciones de viento
        if self.static_wind and self.static_wind_values:
            wind_dir, wind_speed = self.static_wind_values
        else:
            wind_dir, wind_speed = self.wind_field.get(self.position, self.t)

        # Calcular velocidad del barco según el diagrama polar
        relative_angle = (wind_dir - self.heading) % 360
        boat_speed = self.polar_diagram.get_boat_speed(relative_angle, wind_speed)

        # Movimiento en dt minutos
        dt = self.config['dt'] / 60  # convertir a horas
        dx = boat_speed * np.cos(np.radians(self.heading)) * dt
        dy = boat_speed * np.sin(np.radians(self.heading)) * dt
        self.position += np.array([dy, dx])  # latitud y longitud aproximadas

        self.t += self.config['dt']
        done = self.reached_goal(self.position)

        self.state = np.array([
            *self.position,
            self.heading,
            self.speed,
            wind_dir,
            wind_speed
        ], dtype=np.float32)

        reward = compute_reward(self.position, self.goal, boat_speed, wind_dir, self.heading)

        return self.state, reward, done, {}

    def reached_goal(self, pos):
        return np.linalg.norm(pos - self.goal) < self.config['goal_threshold']

    def decode_discrete_action(self, action):
        mapping = {
            0: (-10, -1), 1: (-10, 0), 2: (-10, 1),
            3: (0, -1),   4: (0, 0),   5: (0, 1),
            6: (10, -1),  7: (10, 0),  8: (10, 1),
        }
        return mapping[action]

    def render(self, mode='human'):
        print(f"t={self.t} min | pos={self.position} | heading={self.heading:.1f}° | speed={self.speed:.1f} kn")
