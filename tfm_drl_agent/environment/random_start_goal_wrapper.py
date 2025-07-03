import random
from utils.utils import select_random_start_goal, haversine


class RandomStartGoalWrapper:
    def __init__(self, env_class, config_base, wind_field):
        self.env_class = env_class
        self.config_base = config_base
        self.wind_field = wind_field
        self.available_dates = list(wind_field.wind_data.keys())
        self.env = None
        self.selected_date = None
        self._reset_env()

    def _reset_env(self):
        self.selected_date = random.choice(self.available_dates)
        df_day = self.wind_field.wind_data[self.selected_date]

        # Seleccionar start y goal válidos para ese día
        start, goal = select_random_start_goal(df_day)
        
        dist_km = haversine(start[1], start[0], goal[1], goal[0])
        print(f"[RUTA] Día: {self.selected_date} | Inicio: {start} | Destino: {goal} | Distancia: {dist_km:.2f} km")


        # Construir config
        config = self.config_base.copy()
        config['start'] = start
        config['goal'] = goal
        config['wind'] = self.wind_field.get_day(self.selected_date)
        config['wind_date'] = self.selected_date

        # Construir entorno interno
        self.env = self.env_class(config)

    def reset(self):
        self._reset_env()
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)

    def __getattr__(self, name):
        return getattr(self.env, name)
