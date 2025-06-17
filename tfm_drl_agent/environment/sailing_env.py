import gym
import numpy as np
from gym import spaces

from environment.polar_diagram import get_boat_speed
from environment.wind_model import WindField
from environment.reward import compute_reward


class SailingEnv(gym.Env):
    """
    Entorno personalizado de navegación a vela compatible con OpenAI Gym.
    Estado: [lat, lon, rumbo, velocidad, viento_dir, viento_vel]
    Acción: [cambio_rumbo, cambio_velocidad] (discreto o continuo)
    """
    def __init__(self, config):
        super(SailingEnv, self).__init__()

        self.config = config
        self.wind_field = WindField(config['wind'])
        self.polar_diagram = config['polar_diagram']
        self.grid = config['grid']
        self.goal = config['goal']

        # Espacio de observación
        low = np.array([0, 0, 0, 0, 0, 0])
        high = np.array([1, 1, 360, 20, 360, 20])  # lat, lon, heading, speed, wind_dir, wind_speed
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Espacio de acción
        if config['continuous']:
            self.action_space = spaces.Box(
                low=np.array([-10, -1]), high=np.array([10, 1]), dtype=np.float32  # delta heading, delta speed
            )
        else:
            self.action_space = spaces.Discrete(9)  # 3 heading x 3 speed (e.g. left/none/right x slow/none/fast)

        self.reset()

    def reset(self):
        self.position = np.array(self.config['start'], dtype=np.float32)  # lat, lon
        self.heading = 90.0  # hacia el Este
        self.speed = 5.0     # nudos iniciales
        self.t = 0           # tiempo en minutos

        wind_dir, wind_speed = self.wind_field.get(self.position, self.t)
        self.state = np.array([
            *self.position,
            self.heading,
            self.speed,
            wind_dir,
            wind_speed
        ], dtype=np.float32)

        return self.state

    def step(self, action):
        # Aplicar acción
        if self.config['continuous']:
            delta_heading, delta_speed = action
        else:
            delta_heading, delta_speed = self.decode_discrete_action(action)

        self.heading += delta_heading
        self.speed += delta_speed
        self.heading = self.heading % 360
        self.speed = np.clip(self.speed, 0, 20)

        # Obtener condiciones de viento
        wind_dir, wind_speed = self.wind_field.get(self.position, self.t)

        # Consultar diagrama polar
        relative_wind_angle = (wind_dir - self.heading) % 360
        boat_speed = get_boat_speed(self.polar_diagram, relative_wind_angle, wind_speed)

        # Mover el barco
        dx = boat_speed * np.cos(np.radians(self.heading)) * self.config['dt'] / 60
        dy = boat_speed * np.sin(np.radians(self.heading)) * self.config['dt'] / 60
        self.position += np.array([dy, dx])  # aproximación simple para lat/lon

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
        print(f"t={self.t} pos={self.position} heading={self.heading:.1f} speed={self.speed:.1f}")
