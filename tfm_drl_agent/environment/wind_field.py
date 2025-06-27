import pandas as pd
import numpy as np

class MultiDayWindField:
    def __init__(self, csv_paths_by_date):
        """
        csv_paths_by_date: dict con clave = fecha 'YYYY-MM-DD', valor = ruta al CSV
        Cada CSV debe tener columnas: latitude, longitude, time, wind_speed_10m, wind_direction_10m
        """
        self.wind_data = {}
        for date, path in csv_paths_by_date.items():
            df = pd.read_csv(path)
            df['time'] = pd.to_datetime(df['time'])
            self.wind_data[date] = df

    def get(self, position, t, date):
        """
        Devuelve la dirección y velocidad del viento en la posición y tiempo dados.

        Args:
            position: [lat, lon]
            t: tiempo en minutos desde el inicio del episodio
            date: string 'YYYY-MM-DD'

        Returns:
            wind_dir, wind_speed
        """
        if date not in self.wind_data:
            raise ValueError(f"No hay datos de viento para la fecha {date}")

        df = self.wind_data[date]
        t_min = df['time'].min()
        t_target = t_min + pd.Timedelta(minutes=t)

        # Filtrar por tiempo
        candidates = df[df['time'] == t_target]
        if candidates.empty:
            return 0, 0  # sin datos exactos

        lat, lon = position
        df_close = candidates[
            (abs(candidates['latitude'] - lat) <= 0.05) &
            (abs(candidates['longitude'] - lon) <= 0.05)
        ]

        if df_close.empty:
            return 0, 0  # sin datos cercanos

        df_close = df_close.copy()
        df_close['dist'] = (df_close['latitude'] - lat)**2 + (df_close['longitude'] - lon)**2
        best = df_close.loc[df_close['dist'].idxmin()]

        return best['wind_direction_10m'], best['wind_speed_10m']
