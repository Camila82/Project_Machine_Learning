"""
Servicio de carga y limpieza del dataset.
Se encarga de leer el archivo Excel y aplicar la depuración.
Incluye caché en memoria para evitar releer el archivo en cada petición.
"""
import pandas as pd
import os
from config.settings import DATASET_PATH, COLUMNS_TO_DROP

# ── Caché en memoria ──
_cached_df = None


def load_raw_data():
    """Carga el dataset original sin modificaciones. Usa caché en memoria."""
    global _cached_df

    if _cached_df is not None:
        return _cached_df.copy()

    if not os.path.exists(DATASET_PATH):
        return None

    _cached_df = pd.read_excel(DATASET_PATH)
    return _cached_df.copy()


def clean_data(df):
    """
    Aplica la depuración al DataFrame:
    1. Elimina columnas irrelevantes
    2. Elimina duplicados
    3. Elimina filas con valores nulos
    """
    df_clean = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna()
    return df_clean


def get_original_info(df):
    """Obtiene información del dataset original (antes de limpiar)."""
    return {
        'filas': int(df.shape[0]),
        'columnas': int(df.shape[1]),
        'columnas_lista': df.columns.tolist(),
        'nulos_por_columna': df.isnull().sum().to_dict(),
        'duplicados': int(df.duplicated().sum()),
        'tipos': df.dtypes.astype(str).to_dict(),
    }


def get_cleaned_info(df_clean):
    """Obtiene información del dataset después de la depuración."""
    return {
        'filas': int(df_clean.shape[0]),
        'columnas': int(df_clean.shape[1]),
        'columnas_lista': df_clean.columns.tolist(),
        'nulos_total': int(df_clean.isnull().sum().sum()),
        'duplicados': int(df_clean.duplicated().sum()),
    }
