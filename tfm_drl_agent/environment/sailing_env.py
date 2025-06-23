import numpy as np
import pandas as pd
from gym import Env, spaces
from environment.reward import compute_reward

class SailingEnv(Env):
    def __init__(self, config):
        self.config = config
        self.position = np.array(config['start'])
        self.goal = np.array(config['goal'])
        self.goal_threshold = config.get('goal_threshold', 0.01)
        self.dt = config.get('dt', 10)  # minutos
        self.max_steps = config.get('max_steps', 144)
        self.continuous = config.get('continuous', True)
        self.polar_diagram = config['polar_diagram']
        self.wind_field = config['wind']
        self.t = 0  # tiempo acumulado en minutos
        self.step_count = 0
        self.debug = config.get('debug', False)

        # Estado: [lat, lon, heading, speed, wind_dir, wind_speed]
        low = np.array([-90, -180, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([90, 180, 360, 20, 360, 100], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if self.continuous:
            self.action_space = spaces.Box(low=np.array([-10, -1]), high=np.array([10, 1]), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(9)  # combinaciones de +-10° y +-1 kn

        self.reset()

    def step(self, action):
        self.step_count += 1

        # Acción
        if self.continuous:
            delta_heading, delta_speed = action
        else:
            # Discretizar acciones
            mapping = [(-10, -1), (-10, 0), (-10, 1), (0, -1), (0, 0), (0, 1), (10, -1), (10, 0), (10, 1)]
            delta_heading, delta_speed = mapping[action]

        self.heading = (self.heading + delta_heading) % 360
        self.speed = max(0, self.speed + delta_speed)

        # Obtener viento actual
        wind_dir, wind_speed = self.wind_field.get(self.position, self.t)
        relative_angle = (wind_dir - self.heading) % 360

        # Calcular velocidad del barco según el diagrama polar
        boat_speed = self.polar_diagram.get_speed(relative_angle, wind_speed)

        # Movimiento en dt minutos
        dt = self.dt / 60  # convertir a horas
        dlat = boat_speed * dt * np.cos(np.radians(self.heading)) / 60
        dlon = boat_speed * dt * np.sin(np.radians(self.heading)) / (60 * np.cos(np.radians(self.position[0])))
        self.position += np.array([dlat, dlon])

        self.t += self.dt
        done = self.reached_goal(self.position) or self.t >= 1440 or self.step_count >= self.max_steps

        # Calcular recompensa con debug activado
        reward = compute_reward(self.position, self.goal, boat_speed, wind_dir, self.heading, debug=self.debug)

        # Actualizar estado
        self.state = np.array([
            self.position[0],
            self.position[1],
            self.heading,
            boat_speed,
            wind_dir,
            wind_speed
        ])

        return self.state, reward, done, {}

    def reset(self):
        # Validación opcional: comprobar si duración cabe en datos de viento
        max_time = pd.to_datetime(self.wind_field.df['time'].max())
        t0 = pd.to_datetime(self.wind_field.df['time'].min())
        if t0 + pd.Timedelta(minutes=self.config['max_steps'] * self.config['dt']) > max_time:
            raise ValueError("Duración del episodio excede el rango temporal del viento disponible.")

        self.position = np.array(self.config['start'])
        self.goal = np.array(self.config['goal'])
        self.heading = 0
        self.speed = 0
        self.t = 0
        self.step_count = 0

        wind_dir, wind_speed = self.wind_field.get(self.position, self.t)
        self.state = np.array([
            self.position[0],
            self.position[1],
            self.heading,
            self.speed,
            wind_dir,
            wind_speed
        ])
        return self.state

    def reached_goal(self, pos):
        return np.linalg.norm(pos - self.goal) < self.goal_threshold
