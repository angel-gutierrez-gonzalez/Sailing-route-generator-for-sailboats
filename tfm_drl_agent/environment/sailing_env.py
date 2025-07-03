import gym
from gym import spaces
import numpy as np
import pandas as pd
from environment.reward import compute_reward
import math

class SailingEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wind_field = config['wind']
        self.wind_date = config.get('wind_date', None)
        self.polar_diagram = config['polar_diagram']
        self.dt = config['dt']
        self.max_steps = config['max_steps']
        self.debug = config.get('debug', False)

        # Definir el espacio de acciones (rumbo, velocidad)
        self.continuous = config.get('continuous', True)
        if self.continuous:
            self.action_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([360.0, 20.0]), dtype=np.float32)
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
        if not self.continuous:
            action = self._discrete_to_continuous(action)

        heading, speed = action
        heading = heading % 360
        speed = np.clip(speed, 0, 20)

        lat, lon = self.position
        wind_dir, wind_speed = self.wind_field.get(lat, lon, self.step_count * self.dt)

        # Calcular velocidad del barco según el diagrama polar
        relative_angle = (wind_dir - heading) % 360
        boat_speed = self.polar_diagram.get_speed(relative_angle, wind_speed)

        # Movimiento en dt minutos
        dt = self.config['dt'] / 60  # convertir a horas
        dx = boat_speed * dt * math.cos(math.radians(heading)) / 111  # 1° ~ 111 km
        dy = boat_speed * dt * math.sin(math.radians(heading)) / 111
        self.position += np.array([dy, dx])

        # Calcular recompensa
        reward = compute_reward(self.position, self.goal, boat_speed, wind_dir, heading)

        self.step_count += 1
        self.heading = heading
        self.speed = boat_speed

        done = self._is_done()

        if self.debug:
            print(f"[STEP LOG] dist={np.linalg.norm(self.position - self.goal):.4f}, "
                  f"heel={abs((wind_dir - heading) % 360):.2f}°, "
                  f"angle_diff={abs((heading - self._goal_angle()) % 360):.1f}°, "
                  f"speed={boat_speed:.2f} kn")

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        lat, lon = self.position
        wind_dir, wind_speed = self.wind_field.get(lat, lon, self.step_count * self.dt)
        return np.array([
            lat, lon,
            self.heading, self.speed,
            wind_dir, wind_speed
        ], dtype=np.float32)

    def _is_done(self):
        distance = np.linalg.norm(self.position - self.goal)
        if distance < self.config.get('goal_threshold', 0.01):
            return True
        if self.step_count >= self.max_steps:
            return True
        return False

    def _discrete_to_continuous(self, action_idx):
        # 3x3 acciones: -10°, 0°, +10° y -1, 0, +1 kn desde estado anterior
        delta_heading = [-10, 0, 10]
        delta_speed = [-1, 0, 1]
        i = action_idx // 3
        j = action_idx % 3
        heading = (self.heading + delta_heading[i]) % 360
        speed = np.clip(self.speed + delta_speed[j], 0, 20)
        return np.array([heading, speed], dtype=np.float32)
    
    def _goal_angle(self):
        """
        Devuelve el ángulo (en grados) desde la posición actual hacia el objetivo.
        """
        delta_lat = self.goal[0] - self.position[0]
        delta_lon = self.goal[1] - self.position[1]
        angle_rad = math.atan2(delta_lon, delta_lat)
        return (math.degrees(angle_rad) + 360) % 360

