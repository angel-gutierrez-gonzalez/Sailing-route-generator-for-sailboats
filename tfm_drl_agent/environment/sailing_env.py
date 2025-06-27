import numpy as np
import pandas as pd
from gym import Env, spaces

class SailingEnv(Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.position = np.array(config['start'])
        self.goal = np.array(config['goal'])
        self.dt = config['dt']
        self.max_steps = config['max_steps']
        self.step_count = 0
        self.heading = 0.0
        self.speed = 0.0
        self.polar_diagram = config['polar_diagram']
        self.wind_field = config['wind']
        self.debug = config.get('debug', False)

        if config.get('continuous', True):
            self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(9)  # combinaciones de +-10° y +-1 kn

        # Definir el espacio de observaciones
        self.observation_space = spaces.Box(
            low=np.array([-90, -180, 0, 0, 0, 0]),
            high=np.array([90, 180, 360, 20, 360, 60]),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        # Validación opcional: comprobar si duración cabe en datos de viento
        max_time = pd.to_datetime(self.wind_field.max_time)
        t0 = pd.to_datetime(self.wind_field.min_time)
        if t0 + pd.Timedelta(minutes=self.config['max_steps'] * self.config['dt']) > max_time:
            raise ValueError("Duración del episodio excede el rango temporal del viento disponible.")

        self.position = np.array(self.config['start'])
        self.goal = np.array(self.config['goal'])
        self.step_count = 0
        self.heading = 0.0
        self.speed = 0.0

        return self._get_obs()

    def step(self, action):
        self.step_count += 1

        # Aplicar acción
        delta_heading = action[0] * 10  # hasta +-10°
        delta_speed = action[1] * 1     # hasta +-1 kn
        self.heading = (self.heading + delta_heading) % 360
        self.speed = max(0.1, self.speed + delta_speed)

        # Obtener viento
        wind_dir, wind_speed = self.wind_field.get(self.position, self.step_count * self.dt)

        # Calcular velocidad del barco según el diagrama polar
        relative_angle = (wind_dir - self.heading) % 360
        boat_speed = self.polar_diagram.get_speed(relative_angle, wind_speed)

        # Movimiento en dt minutos
        dt = self.config['dt'] / 60  # convertir a horas
        dx = boat_speed * dt * np.cos(np.radians(self.heading))
        dy = boat_speed * dt * np.sin(np.radians(self.heading))
        self.position += np.array([dx, dy])

        # Calcular recompensa
        from environment.reward import compute_reward
        reward = compute_reward(self.position, self.goal, boat_speed, wind_dir, self.heading, debug=self.debug)

        # Comprobar fin de episodio
        done = self._reached_goal(self.position) or self.step_count >= self.max_steps
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        wind_dir, wind_speed = self.wind_field.get(self.position, self.step_count * self.dt)
        return np.array([
            self.position[0], self.position[1],
            self.heading, self.speed,
            wind_dir, wind_speed
        ], dtype=np.float32)

    def _reached_goal(self, pos):
        return np.linalg.norm(pos - self.goal) < self.config.get('goal_threshold', 0.01)
