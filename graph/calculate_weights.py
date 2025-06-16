import math
import pandas as pd


class PolarDiagram:
    """
    Representa un diagrama polar de veleros (TWA vs Speed).
    Se inicializa con un DataFrame que contiene columnas 'TWA' y 'Speed'.
    """
    def __init__(self, polar_df: pd.DataFrame, twa_col: str = 'TWA', speed_col: str = None):
        df = polar_df.copy()
        # Validar columna de TWA
        if twa_col not in df.columns:
            raise KeyError(f"TWA column '{twa_col}' no encontrada, disponibles: {df.columns.tolist()}")
        # Detección de columna de velocidad
        if speed_col is None:
            candidates = [c for c in df.columns if 'speed' in c.lower()]
            if not candidates:
                raise KeyError(f"No se encontró columna de velocidad, disponibles: {df.columns.tolist()}")
            speed_col = candidates[0]
        df = df.rename(columns={twa_col: 'TWA', speed_col: 'Speed'})
        self.df = df[['TWA', 'Speed']].sort_values('TWA').reset_index(drop=True)

    def get_speed(self, twa: float) -> float:
        """
        Dada una verdadera ángulo de viento (TWA), interpola en el diagrama polar
        y devuelve la velocidad correspondiente.
        """
        # Normalizar TWA al rango [-180,180]
        twa_norm = abs(((twa + 180) % 360) - 180)
        df = self.df
        # Si fuera de los límites, devolver extremo
        if twa_norm <= df['TWA'].iloc[0]:
            return df['Speed'].iloc[0]
        if twa_norm >= df['TWA'].iloc[-1]:
            return df['Speed'].iloc[-1]
        # Encontrar bracketing
        lower = df[df['TWA'] <= twa_norm].iloc[-1]
        upper = df[df['TWA'] >= twa_norm].iloc[0]
        if lower['TWA'] == upper['TWA']:
            return lower['Speed']
        # Interpolación lineal
        frac = (twa_norm - lower['TWA']) / (upper['TWA'] - lower['TWA'])
        return lower['Speed'] + frac * (upper['Speed'] - lower['Speed'])
