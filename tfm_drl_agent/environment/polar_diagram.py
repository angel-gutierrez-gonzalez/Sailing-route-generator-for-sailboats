import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

class PolarDiagram:
    """
    Clase para consultar la velocidad del barco según el ángulo de viento (TWA) y la velocidad del viento (TWS).
    Acepta un CSV en formato ancho con separador ';', donde:
      - La primera columna son los TWA (ángulos de viento)
      - Las columnas son TWS (velocidades del viento)
    """
    def __init__(self, csv_path):
        # Leer CSV ancho con separador ';'
        df_pivot = pd.read_csv(csv_path, sep=';', index_col=0)

        # Asegurar orden creciente y conversión a float
        df_pivot.columns = df_pivot.columns.astype(float)
        df_pivot.index = df_pivot.index.astype(float)
        self.df = df_pivot.sort_index().sort_index(axis=1)

        # Extraer ejes y valores
        self.angles = np.sort(self.df.index.values.astype(float))             # TWA
        self.wind_speeds = np.sort(self.df.columns.astype(float).values)      # TWS
        self.values = self.df.values.T.astype(float)                           # matriz TWS x TWA

        # Crear interpolador 2D
        self.interpolator = RegularGridInterpolator(
            (self.wind_speeds, self.angles), self.values, bounds_error=False, fill_value=None
        )

    def get_speed(self, twa_deg, tws):
        """
        Devuelve la velocidad del barco para un ángulo TWA y una velocidad de viento TWS.

        Args:
            twa_deg: True Wind Angle (0-180°)
            tws: True Wind Speed (nudos)

        Returns:
            Velocidad interpolada del barco (nudos)
        """
        twa_clamped = np.clip(twa_deg, min(self.angles), max(self.angles))
        tws_clamped = np.clip(tws, min(self.wind_speeds), max(self.wind_speeds))
        speed = self.interpolator([[tws_clamped, twa_clamped]])[0]
        return speed
