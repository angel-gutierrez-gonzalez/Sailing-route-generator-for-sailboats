import pandas as pd

class SingleDayWind:
    def __init__(self, df):
        self.df = df.copy()
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.min_time = self.df['time'].min()
        self.max_time = self.df['time'].max()

    def get(self, lat, lon, t_target):
        candidates = self.df[self.df['time'] == t_target]
        if candidates.empty:
            return 0.0, 0.0

        distances = ((candidates['latitude'] - lat) ** 2 + (candidates['longitude'] - lon) ** 2)
        nearest_idx = distances.idxmin()

        if nearest_idx not in candidates.index:
            return 0.0, 0.0

        nearest = candidates.loc[nearest_idx]
        return nearest['wind_direction_10m'], nearest['wind_speed_10m']


class MultiDayWindField:
    def __init__(self, csv_paths_by_date):
        """
        csv_paths_by_date: dict con clave = fecha 'YYYY-MM-DD', valor = ruta al CSV
        Cada CSV debe tener columnas: latitude, longitude, time, wind_speed_10m, wind_direction_10m
        """
        self.wind_data = {}
        self.min_time = {}
        self.max_time = {}

        for date, path in csv_paths_by_date.items():
            df = pd.read_csv(path)
            df = df[df['time'].notna()].copy()
            df['time'] = pd.to_datetime(df['time'])
            self.wind_data[date] = df
            self.min_time[date] = df['time'].min()
            self.max_time[date] = df['time'].max()

    def get_day(self, date):
        """
        Devuelve un SingleDayWind para la fecha especificada.
        """
        if date not in self.wind_data:
            raise ValueError(f"No hay datos de viento para la fecha {date}")
        return SingleDayWind(self.wind_data[date])
