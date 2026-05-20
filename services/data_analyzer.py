"""
Servicio de análisis y estadísticas del dataset.
Genera estadísticas descriptivas, distribuciones y datos de vista previa.
"""
from config.settings import NUMERIC_COLUMNS, PREVIEW_COLUMNS, PREVIEW_ROWS


def get_descriptive_stats(df):
    """Calcula estadísticas descriptivas para las columnas numéricas."""
    stats = {}
    for col in NUMERIC_COLUMNS:
        stats[col] = {
            'media': round(float(df[col].mean()), 2),
            'mediana': round(float(df[col].median()), 2),
            'desv_std': round(float(df[col].std()), 2),
            'min': round(float(df[col].min()), 2),
            'max': round(float(df[col].max()), 2),
        }
    return stats


def get_target_distribution(df):
    """Obtiene la distribución de la variable objetivo."""
    return df['Clasificacion_Normativa'].value_counts().to_dict()


def get_city_distribution(df):
    """Calcula el nivel promedio de ruido por ciudad."""
    return df.groupby('Ciudad')['Nivel_Ruido_dB'].mean().round(1).to_dict()


def get_preview_data(df):
    """Obtiene una muestra de datos para la tabla de vista previa."""
    return df[PREVIEW_COLUMNS].head(PREVIEW_ROWS).to_dict(orient='records')
