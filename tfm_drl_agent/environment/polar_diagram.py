import numpy as np
import pandas as pd
from scipy.interpolate import interp2d

class PolarDiagram:
    """
    Clase para consultar velocidad del barco según ángulo de viento y velocidad de viento.
    Carga datos desde un CSV tipo: filas = wind speed (nudos), columnas = TWA (True Wind Angle).
    """
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.wind_speeds = self.df.index.values.astype(float)             # Ej: [4, 6, 8, 10, 12, ...]
        self.angles = self.df.columns.astype(float).values               # Ej: [0, 15, 30, ..., 180]
        self.values = self.df.values.astype(float)                       # matriz de velocidades

        # Crear interpolador 2D
        self.interpolator = interp2d(self.angles, self.wind_speeds, self.values, kind='linear')

    def get_boat_speed(self, twa_deg, tws):
        """
        Args:
            twa_deg: True Wind Angle en grados (0-180)
            tws: True Wind Speed en nudos

        Returns:
            Velocidad del barco en nudos
        """
        twa_clamped = np.clip(twa_deg, min(self.angles), max(self.angles))
        tws_clamped = np.clip(tws, min(self.wind_speeds), max(self.wind_speeds))
        speed = self.interpolator(twa_clamped, tws_clamped)[0]
        return speed
