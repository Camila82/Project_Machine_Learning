import json
import os
from pathlib import Path

# Obtener la ruta absoluta a la carpeta de datos procesados
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

def load_json_artifact(filename):
    """
    Carga un archivo JSON desde la carpeta data/processed/.
    Retorna un diccionario vacío si el archivo no existe para evitar que Flask colapse.
    """
    file_path = PROCESSED_DIR / filename
    
    if not os.path.exists(file_path):
        print(f"Warning: Artifact {filename} not found at {file_path}")
        return {}
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return {}

def get_all_data_artifacts():
    """Carga todos los artefactos necesarios para la Fase 1."""
    return {
        "general": load_json_artifact('stats_general.json'),
        "eda": load_json_artifact('stats_eda.json'),
        "correlations": load_json_artifact('stats_correlations.json'),
        "by_zone": load_json_artifact('stats_by_zone.json'),
        "sample": load_json_artifact('dataset_sample.json')
    }