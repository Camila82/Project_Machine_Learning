"""
Configuration settings for the application.
Centralizes file paths, column names, and project parameters.
"""
import os

# ── File Paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
DATASET_PATH = os.path.join(DATABASE_DIR, 'dataset_ruido_urbano_colombia.csv')

# ── Columns to drop during cleaning ──
COLUMNS_TO_DROP = ['ID_Registro', 'Observaciones', 'Latitud', 'Longitud', 'Fecha']

# ── Numeric columns for descriptive statistics ──
NUMERIC_COLUMNS = [
    'Nivel_Ruido_dB',
    'Flujo_Vehicular_veh_h',
    'Velocidad_Promedio_kmh',
    'Saturacion_Transporte',
    'Hora',
    'Indice_Movilidad',
]

# ── Columns for data preview table ──
PREVIEW_COLUMNS = [
    'Ciudad', 'Hora', 'Tipo_Zona', 'Fuente_Principal_Ruido',
    'Nivel_Ruido_dB', 'Clasificacion_Normativa', 'Flujo_Vehicular_veh_h',
]

PREVIEW_ROWS = 15

# ── Flask Settings ──
FLASK_PORT = 5001
FLASK_DEBUG = True
