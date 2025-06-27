import gym
import random

class RandomizedWindWrapper(gym.Wrapper):
    def __init__(self, env_class, config_base, wind_field):
        """
        Wrapper que reinicializa el entorno con un viento diferente cada vez (por fecha).

        Args:
            env_class: clase del entorno base (e.g. SailingEnv)
            config_base: diccionario base de configuración sin 'wind'
            wind_field: instancia de MultiDayWindField
        """
        self.env_class = env_class
        self.config_base = config_base
        self.wind_field = wind_field
        self.available_dates = list(wind_field.wind_data.keys())
        self.env = self._reset_env()  # Inicializar entorno
        super().__init__(self.env)

    def _reset_env(self):
        date = random.choice(self.available_dates)
        print(f"[RANDOMIZED WIND] Episodio con condiciones del día: {date}")
        config = self.config_base.copy()
        config['wind'] = self._wrap_date(date)
        return self.env_class(config)

    def reset(self):
        self.env = self._reset_env()
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def _wrap_date(self, date):
        """Clase interna para acceder a viento de un solo día."""
        class SingleDayWind:
            def __init__(self, base, fixed_date):
                self.base = base
                self.date = fixed_date

            def get(self, position, t):
                return self.base.get(position, t, self.date)

            @property
            def max_time(self):
                return self.base.wind_data[self.date]['time'].max()

            @property
            def min_time(self):
                return self.base.wind_data[self.date]['time'].min()

        return SingleDayWind(self.wind_field, date)
