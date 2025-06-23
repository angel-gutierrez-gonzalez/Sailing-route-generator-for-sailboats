import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WindField:
    """
    Modelo de viento espacial-temporal construido a partir de datos obtenidos de Open-Meteo.
    Requiere un DataFrame con columnas: latitude, longitude, time, wind_speed_10m, wind_direction_10m.
    """
    def __init__(self, df_wind):
        self.df = df_wind.copy()
        self.df['time'] = pd.to_datetime(self.df['time'])

    def get(self, position, t_min):
        """
        Obtiene viento interpolado para una posicion y tiempo dado.

        Args:
            position: [lat, lon]
            t_min: tiempo en minutos desde t0 (se espera UTC)

        Returns:
            wind_dir (grados), wind_speed (nudos)
        """
        target_time = self.df['time'].min() + timedelta(minutes=t_min)

        # Buscar los puntos cercanos en lat/lon
        lat, lon = position
        df_area = self.df[
            (self.df['latitude'].between(lat - 0.05, lat + 0.05)) &
            (self.df['longitude'].between(lon - 0.05, lon + 0.05))
        ]

        if df_area.empty:
            return 0.0, 0.0  # Viento nulo si fuera de cobertura

        # Tomar valores más cercanos en tiempo
        times = df_area['time'].unique()
        closest_time = min(times, key=lambda t: abs(t - target_time))

        df_t = df_area[df_area['time'] == closest_time]
        df_t = df_t.copy()
        df_t['dist'] = np.sqrt((df_t['latitude'] - lat)**2 + (df_t['longitude'] - lon)**2)
        nearest = df_t.loc[df_t['dist'].idxmin()]

        return nearest['wind_direction_10m'], nearest['wind_speed_10m']
