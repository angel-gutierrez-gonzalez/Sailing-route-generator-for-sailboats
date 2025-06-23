import numpy as np
import pandas as pd
from scipy.interpolate import interp2d

class PolarDiagram:
    def __init__(self, polar_df, twa_col='TWA', speed_col=None):
        df = polar_df.copy()
        # Renombrar columnas
        if twa_col not in df.columns:
            raise KeyError(f"TWA column '{twa_col}' not found")
        if speed_col is None:
            candidates = [c for c in df.columns if 'speed' in c.lower()]
            if not candidates:
                raise KeyError("No speed column found")
            speed_col = candidates[0]
        df = df.rename(columns={twa_col:'TWA', speed_col:'Speed'})
        # Quitar duplicados y ordenar
        df = df[['TWA','Speed']].drop_duplicates('TWA').sort_values('TWA').reset_index(drop=True)
        self.twas = df['TWA'].values
        self.speeds = df['Speed'].values

    def get_speed(self, twa: float) -> float:
        """
        Devuelve gamma(w, beta) interpolado en el diagrama polar.
        Extrapola en los extremos sin IndexError.
        """
        # Normalizar twa a [-180,180]
        twa = abs(((twa + 180) % 360) - 180)
        # Interpolar linealmente, extrapolando a los extremos
        return float(np.interp(twa, self.twas, self.speeds))