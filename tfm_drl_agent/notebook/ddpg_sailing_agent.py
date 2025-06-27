# Configuración e imports

import sys
import os
import tensorflow as tf

# Obtener ruta absoluta del directorio que contiene el notebook
notebook_dir = os.path.dirname(os.getcwd())  # sube un nivel desde /notebook
if notebook_dir not in sys.path:
    sys.path.append(notebook_dir)

import pandas as pd
from environment.sailing_env import SailingEnv
from environment.wind_field import MultiDayWindField
from environment.random_wind_wrapper import RandomizedWindWrapper
from environment.polar_diagram import PolarDiagram
from agents.ddpg import DDPGAgent
from training.train_drl import train_ddpg

# Carga de datos

# In[6]:


# Diagrama polar
polar = PolarDiagram('../../data/polar_diagram.csv')

# Carpeta de rutas CSV del profesor
folder_path = "../../data/expert_trajectories"

# Definir rutas por fecha (ajusta con tus archivos)
csv_paths = {
    "2025-05-04": "../../data/processed/nodes_bathy_wind20250504.csv",
    "2025-05-04": "../../data/processed/nodes_bathy_wind20250507.csv",
    "2025-05-09": "../../data/processed/nodes_bathy_wind20250509.csv",
    "2025-05-11": "../../data/processed/nodes_bathy_wind20250511.csv",
    "2025-05-13": "../../data/processed/nodes_bathy_wind20250513.csv",
    "2025-05-15": "../../data/processed/nodes_bathy_wind20250515.csv",
    "2025-05-17": "../../data/processed/nodes_bathy_wind20250517.csv",
    "2025-05-19": "../../data/processed/nodes_bathy_wind20250519.csv",
    "2025-05-23": "../../data/processed/nodes_bathy_wind20250523.csv",
    "2025-05-25": "../../data/processed/nodes_bathy_wind20250525.csv",
    "2025-05-27": "../../data/processed/nodes_bathy_wind20250527.csv",
    "2025-05-28": "../../data/processed/nodes_bathy_wind20250528.csv",
    "2025-05-30": "../../data/processed/nodes_bathy_wind20250530.csv",
    "2025-06-01": "../../data/processed/nodes_bathy_wind20250601.csv"
}

# Crear campo de viento multi-día
wind_field = MultiDayWindField(csv_paths)


# Inicialización del entorno

# - El barco empieza y termina dentro del area navegable.
# - El episodio dura como maximo 24 horas.
# - Las consultas de viento estan cubiertas por tu CSV de Open-Meteo.

# Configuración de entorno simulado
config_base = {
    "start": [38.5, 1.0],
    "goal": [40.0, 4.5],
    "dt": 10,
    "max_steps": 135,
    "polar_diagram": polar,
    "debug": True
}

# Crear entorno con viento que cambia por episodio
env = RandomizedWindWrapper(SailingEnv, config_base, wind_field)


# Creación del agente

agent = DDPGAgent(env)


# Entrenamiento

# Entrenamiento con aprendizaje por imitación

from agents.ddpg import get_actor
from training.imitation_learning import train_actor_supervised

# Crear actor sin entrenar
actor_model = get_actor(input_shape=6, action_bounds=[360, 20])

# Entrenar actor con todas las rutas disponibles
trained_actor = train_actor_supervised(actor_model, folder_path, epochs=200)


# Entrenamiento por refuerzo (Entorno simulado)
rewards = train_ddpg(agent, env, episodes=700)
