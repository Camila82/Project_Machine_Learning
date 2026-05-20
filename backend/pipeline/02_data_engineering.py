"""
==============================================================================
CRISP-ML | FASE 2: INGENIERÍA DE DATOS
Proyecto : Sistema de Análisis de Ruido Urbano - Colombia
Autor    : Arquitectura MLOps Senior / Steven Naranjo
Versión  : 1.0.1 (Fix: Excel Source & Column Cleansing)
==============================================================================
"""

import os
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL DOMINIO
# ─────────────────────────────────────────────────────────────────────────────

# Rutas del proyecto
ROOT_DIR = Path(__file__).resolve().parents[2]
# CORRECCIÓN: Apuntamos al archivo real .xlsx que tienes en tu carpeta data/raw
RAW_PATH = ROOT_DIR / "data" / "raw" / "dataset_ruido_urbano_colombia.xlsx"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Variables numéricas del dominio acústico y de movilidad
NUMERIC_FEATURES = [
    "Nivel_Ruido_dB",
    "Indice_Movilidad",
    "Flujo_Vehicular_veh_h",
    "Velocidad_Promedio_kmh",
    "Saturacion_Transporte",
    "Calidad_Señal_Sensor",
]

# Variables categóricas a codificar para el modelo ML
CATEGORICAL_FEATURES = [
    "Ciudad",
    "Tipo_Zona",
    "Tipo_Sensor",
    "Fuente_Principal_Ruido",
    "Condicion_Climatica",
]

# Variable objetivo: clasificación normativa del nivel de ruido
TARGET_COLUMN = "Clasificacion_Normativa"

# Orden lógico de la variable objetivo (para encoding ordinal)
TARGET_ORDER = ["Bajo", "Moderado", "Alto", "Crítico"]

# Franjas horarias según normativa colombiana de ruido (Resolución 627/2006)
FRANJAS_HORARIAS = {
    "Madrugada": range(0, 6),    # 00:00 - 05:59  → menor tolerancia
    "Mañana":    range(6, 12),   # 06:00 - 11:59
    "Tarde":     range(12, 18),  # 12:00 - 17:59
    "Noche":     range(18, 24),  # 18:00 - 23:59  → mayor impacto social
}

# Límites de ruido por tipo de zona (dB) - Resolución 627/2006 MADS Colombia
LIMITES_NORMATIVOS = {
    "Residencial":  {"diurno": 65, "nocturno": 55},
    "Comercial":    {"diurno": 70, "nocturno": 60},
    "Industrial":   {"diurno": 75, "nocturno": 70},
    "Hospitalaria": {"diurno": 55, "nocturno": 45},
    "Educativa":    {"diurno": 65, "nocturno": 55},
    "Mixta":        {"diurno": 68, "nocturno": 58},
}


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: CARGA Y VALIDACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def cargar_y_validar(ruta: Path) -> pd.DataFrame:
    """
    Carga el Excel crudo, normaliza nombres de columnas y ejecuta la auditoría.
    """
    log.info("=" * 60)
    log.info("PASO 1: CARGA Y AUDITORÍA DE CALIDAD")
    log.info("=" * 60)

    # CORRECCIÓN: Leemos usando read_excel debido a la extensión .xlsx del origen
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta}")
        
    df = pd.read_excel(ruta)
    
    # CORRECCIÓN INTERNA: Limpieza de espacios en blanco en los nombres de columnas
    df.columns = df.columns.str.strip()
    log.info(f"[AUDITORÍA] Columnas detectadas: {list(df.columns)}")
    
    # Validar y parsear la columna de tiempo de manera segura
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"])
    elif "fecha" in df.columns:
        df.rename(columns={"fecha": "Fecha"}, inplace=True)
        df["Fecha"] = pd.to_datetime(df["Fecha"])
    else:
        raise ValueError("Error crítico: No se encontró la columna 'Fecha' (o 'fecha') en el archivo.")

    log.info(f"Dataset cargado correctamente → {df.shape[0]} registros × {df.shape[1]} columnas")

    # ── Nulos ────────────────────────────────────────────────────────────────
    nulos = df.isnull().sum()
    nulos_totales = nulos.sum()
    if nulos_totales == 0:
        log.info("✓ Integridad: 0 valores nulos en todas las columnas")
    else:
        log.warning(f"⚠ Valores nulos detectados, aplicando imputación básica:\n{nulos[nulos > 0]}")
        # Tratamiento preventivo rápido para que no rompa análisis posteriores
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)

    # ── Duplicados ───────────────────────────────────────────────────────────
    if "ID_Registro" in df.columns:
        duplicados = df.duplicated(subset=["ID_Registro"]).sum()
        if duplicados == 0:
            log.info("✓ Unicidad: 0 IDs de registro duplicados")
        else:
            log.warning(f"⚠ {duplicados} registros con ID_Registro duplicado. Removiendo...")
            df.drop_duplicates(subset=["ID_Registro"], keep="first", inplace=True)
    else:
        log.info("ℹ Nota: No se detectó columna 'ID_Registro', se asume unicidad por índice.")

    # ── Outliers en variable acústica principal ───────────────────────────────
    q1 = df["Nivel_Ruido_dB"].quantile(0.25)
    q3 = df["Nivel_Ruido_dB"].quantile(0.75)
    iqr = q3 - q1
    outliers = df[
        (df["Nivel_Ruido_dB"] < q1 - 1.5 * iqr) |
        (df["Nivel_Ruido_dB"] > q3 + 1.5 * iqr)
    ]
    log.info(
        f"ℹ Nivel_Ruido_dB → min={df['Nivel_Ruido_dB'].min():.1f} | "
        f"max={df['Nivel_Ruido_dB'].max():.1f} | "
        f"media={df['Nivel_Ruido_dB'].mean():.2f} | "
        f"outliers IQR={len(outliers)}"
    )

    # ── Resumen de distribución del target ───────────────────────────────────
    log.info(f"ℹ Distribución objetivo:\n{df[TARGET_COLUMN].value_counts().to_string()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: INGENIERÍA DE CARACTERÍSTICAS
# ─────────────────────────────────────────────────────────────────────────────

def ingenieria_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas variables derivadas con valor predictivo y de negocio.
    """
    log.info("=" * 60)
    log.info("PASO 2: INGENIERÍA DE CARACTERÍSTICAS")
    log.info("=" * 60)

    df = df.copy()

    # ── Feature 1: Franja horaria (negocio + normativo) ───────────────────────
    def asignar_franja(hora: int) -> str:
        for franja, rango in FRANJAS_HORARIAS.items():
            if hora in rango:
                return franja
        return "Noche"

    df["Franja_Horaria"] = df["Hora"].apply(asignar_franja)
    log.info(f"✓ Franja_Horaria creada → {df['Franja_Horaria'].value_counts().to_dict()}")

    # ── Feature 2: Es hora pico (7-9h y 17-19h, patrón colombiano) ───────────
    horas_pico = list(range(7, 10)) + list(range(17, 20))
    df["Es_Hora_Pico"] = df["Hora"].isin(horas_pico).astype(int)
    log.info(f"✓ Es_Hora_Pico → pico={df['Es_Hora_Pico'].sum()} | no pico={len(df)-df['Es_Hora_Pico'].sum()}")

    # ── Feature 3: Excede límite normativo colombiano ─────────────────────────
    def supera_limite(row) -> int:
        zona = row["Tipo_Zona"]
        db   = row["Nivel_Ruido_dB"]
        franja = row["Franja_Horaria"]
        if zona not in LIMITES_NORMATIVOS:
            return 0
        limite = LIMITES_NORMATIVOS[zona]["nocturno"] if franja in ["Madrugada", "Noche"] else LIMITES_NORMATIVOS[zona]["diurno"]
        return int(db > limite)

    df["Supera_Limite_Normativo"] = df.apply(supera_limite, axis=1)
    pct_supera = df["Supera_Limite_Normativo"].mean() * 100
    log.info(f"✓ Supera_Limite_Normativo → {df['Supera_Limite_Normativo'].sum()} registros ({pct_supera:.1f}%)")

    # ── Feature 4: Índice de congestión combinado ────────────────────────────
    df["Indice_Congestion"] = (
        df["Saturacion_Transporte"] * 0.6 +
        (df["Indice_Movilidad"] / 100) * 0.4
    ).round(4)
    log.info(f"✓ Indice_Congestion → media={df['Indice_Congestion'].mean():.3f} | max={df['Indice_Congestion'].max():.3f}")

    # ── Feature 5: Ratio ruido/velocidad (proxy de impacto en circulación) ───
    df["Ratio_Ruido_Velocidad"] = (
        df["Nivel_Ruido_dB"] / df["Velocidad_Promedio_kmh"].replace(0, np.nan)
    ).fillna(0).round(4)
    log.info(f"✓ Ratio_Ruido_Velocidad → media={df['Ratio_Ruido_Velocidad'].mean():.3f}")

    # ── Feature 6: Período del día (binario diurno/nocturno) ──────────────────
    df["Es_Diurno"] = df["Hora"].between(6, 21).astype(int)
    log.info(f"✓ Es_Diurno → diurno={df['Es_Diurno'].sum()} | nocturno={(df['Es_Diurno']==0).sum()}")

    # ── Feature 7: Mes y trimestre (estacionalidad) ───────────────────────────
    df["Mes"]       = df["Fecha"].dt.month
    df["Trimestre"] = df["Fecha"].dt.quarter
    log.info(f"✓ Mes y Trimestre extraídos de Fecha")

    log.info(f"→ Total features generadas: 7 nuevas columnas | Shape final: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: PREPARACIÓN PARA ML (Codificación + Normalización)
# ─────────────────────────────────────────────────────────────────────────────

def preparar_para_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera el dataset ML-ready.
    """
    log.info("=" * 60)
    log.info("PASO 3: CODIFICACIÓN Y NORMALIZACIÓN PARA ML")
    log.info("=" * 60)

    df_ml = df.copy()

    # ── Encoding ordinal del target ───────────────────────────────────────────
    target_map = {label: idx for idx, label in enumerate(TARGET_ORDER)}
    df_ml[f"{TARGET_COLUMN}_encoded"] = df_ml[TARGET_COLUMN].map(target_map)
    log.info(f"✓ Target encoding → {target_map}")

    # ── Label encoding de categóricas ────────────────────────────────────────
    encoding_map = {}
    all_categoricals = CATEGORICAL_FEATURES + ["Franja_Horaria"]

    for col in all_categoricals:
        le = LabelEncoder()
        df_ml[f"{col}_enc"] = le.fit_transform(df_ml[col].astype(str))
        encoding_map[col] = dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))
        log.info(f"✓ LabelEncoding {col} → {encoding_map[col]}")

    # ── MinMax scaling de numéricas ───────────────────────────────────────────
    scaler = MinMaxScaler()
    numeric_to_scale = NUMERIC_FEATURES + ["Indice_Congestion", "Ratio_Ruido_Velocidad"]
    scaled_cols = [f"{c}_scaled" for c in numeric_to_scale]

    df_ml[scaled_cols] = scaler.fit_transform(df_ml[numeric_to_scale])
    log.info(f"✓ MinMaxScaler aplicado a {len(numeric_to_scale)} variables numéricas")

    # ── Persistir mapas de encoding para la API ──────────────────────────────
    encoding_artifact = {
        "target_map": target_map,
        "target_order": TARGET_ORDER,
        "categorical_encodings": encoding_map,
        "numeric_features_scaled": numeric_to_scale,
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
    }
    artifact_path = PROCESSED_DIR / "encoding_map.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(encoding_artifact, f, ensure_ascii=False, indent=2)
    log.info(f"✓ Artefacto de encoding guardado → {artifact_path.name}")

    return df_ml


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4: ESTADÍSTICAS PARA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def generar_stats_dashboard(df: pd.DataFrame) -> None:
    """
    Calcula y exporta todos los agregados JSON que consumirá la interfaz.
    """
    log.info("=" * 60)
    log.info("PASO 4: GENERANDO ESTADÍSTICAS PARA DASHBOARD")
    log.info("=" * 60)

    # ── 4.1 KPIs Generales ───────────────────────────────────────────────────
    stats_generales = {
        "total_registros": int(len(df)),
        "ciudades_cubiertas": int(df["Ciudad"].nunique()),
        "rango_fechas": {
            "inicio": str(df["Fecha"].min().date()),
            "fin":    str(df["Fecha"].max().date()),
        },
        "nivel_ruido": {
            "minimo":  round(float(df["Nivel_Ruido_dB"].min()), 1),
            "maximo":  round(float(df["Nivel_Ruido_dB"].max()), 1),
            "promedio": round(float(df["Nivel_Ruido_dB"].mean()), 2),
            "mediana":  round(float(df["Nivel_Ruido_dB"].median()), 2),
            "std":      round(float(df["Nivel_Ruido_dB"].std()), 2),
        },
        "pct_supera_normativa": round(float(df["Supera_Limite_Normativo"].mean() * 100), 2),
        "pct_hora_pico":        round(float(df["Es_Hora_Pico"].mean() * 100), 2),
        "distribucion_clasificacion": df[TARGET_COLUMN].value_counts().to_dict(),
        "distribucion_fuente":  df["Fuente_Principal_Ruido"].value_counts().to_dict(),
        "distribucion_sensor":  df["Tipo_Sensor"].value_counts().to_dict(),
    }
    _guardar_json(stats_generales, "stats_generales.json")

    # ── 4.2 Estadísticas por Ciudad ──────────────────────────────────────────
    stats_ciudad = {}
    for ciudad, grupo in df.groupby("Ciudad"):
        stats_ciudad[ciudad] = {
            "total_mediciones":      int(len(grupo)),
            "nivel_ruido_promedio":  round(float(grupo["Nivel_Ruido_dB"].mean()), 2),
            "nivel_ruido_max":       round(float(grupo["Nivel_Ruido_dB"].max()), 1),
            "pct_critico":           round(float((grupo[TARGET_COLUMN] == "Crítico").mean() * 100), 2),
            "pct_supera_normativa":  round(float(grupo["Supera_Limite_Normativo"].mean() * 100), 2),
            "indice_movilidad_prom": round(float(grupo["Indice_Movilidad"].mean()), 2),
            "flujo_vehicular_prom":  round(float(grupo["Flujo_Vehicular_veh_h"].mean()), 1),
            "clasificacion_dist":    grupo[TARGET_COLUMN].value_counts().to_dict(),
            "fuente_dist":           grupo["Fuente_Principal_Ruido"].value_counts().to_dict(),
            "latitud_centro":        round(float(grupo["Latitud"].mean()), 4) if "Latitud" in grupo.columns else 0.0,
            "longitud_centro":       round(float(grupo["Longitud"].mean()), 4) if "Longitud" in grupo.columns else 0.0,
        }
    _guardar_json(stats_ciudad, "stats_por_ciudad.json")

    # ── 4.3 Estadísticas por Tipo de Zona ────────────────────────────────────
    stats_zona = {}
    for zona, grupo in df.groupby("Tipo_Zona"):
        limite_diurno  = LIMITES_NORMATIVOS.get(zona, {}).get("diurno", "N/A")
        limite_nocturno = LIMITES_NORMATIVOS.get(zona, {}).get("nocturno", "N/A")
        stats_zona[zona] = {
            "total_mediciones":     int(len(grupo)),
            "nivel_ruido_promedio": round(float(grupo["Nivel_Ruido_dB"].mean()), 2),
            "nivel_ruido_max":      round(float(grupo["Nivel_Ruido_dB"].max()), 1),
            "pct_supera_normativa": round(float(grupo["Supera_Limite_Normativo"].mean() * 100), 2),
            "limite_diurno_dB":     limite_diurno,
            "limite_nocturno_dB":   limite_nocturno,
            "clasificacion_dist":   grupo[TARGET_COLUMN].value_counts().to_dict(),
        }
    _guardar_json(stats_zona, "stats_por_zona.json")

    # ── 4.4 Distribución Horaria ──────────────────────────────────────────────
    stats_hora = {}
    for hora, grupo in df.groupby("Hora"):
        stats_hora[int(hora)] = {
            "total_mediciones":     int(len(grupo)),
            "nivel_ruido_promedio": round(float(grupo["Nivel_Ruido_dB"].mean()), 2),
            "nivel_ruido_max":      round(float(grupo["Nivel_Ruido_dB"].max()), 1),
            "franja":               grupo["Franja_Horaria"].iloc[0],
            "pct_critico":          round(float((grupo[TARGET_COLUMN] == "Crítico").mean() * 100), 2),
        }
    _guardar_json(stats_hora, "stats_por_hora.json")

    # ── 4.5 Correlaciones numéricas ───────────────────────────────────────────
    cols_corr = NUMERIC_FEATURES + ["Indice_Congestion", "Ratio_Ruido_Velocidad",
                                     "Es_Hora_Pico", "Es_Diurno", "Supera_Limite_Normativo"]
    corr_matrix = df[cols_corr].corr().round(4)
    stats_corr = {
        "columnas": cols_corr,
        "matriz":   corr_matrix.values.tolist(),
        "corr_con_ruido": corr_matrix["Nivel_Ruido_dB"].sort_values(ascending=False).to_dict(),
    }
    _guardar_json(stats_corr, "stats_correlaciones.json")

    log.info("✓ Todos los archivos JSON de estadísticas generados correctamente")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITARIO: Guardar JSON
# ─────────────────────────────────────────────────────────────────────────────

def _guardar_json(data: dict, nombre: str) -> None:
    ruta = PROCESSED_DIR / nombre
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"  → Guardado: {nombre}")


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  CRISP-ML | FASE 2 — INGENIERÍA DE DATOS                ║")
    log.info("║  Ruido Urbano Colombia — Pipeline de Procesamiento       ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    # Paso 1: Carga y validación
    df_raw = cargar_y_validar(RAW_PATH)

    # Paso 2: Feature engineering
    df_clean = ingenieria_caracteristicas(df_raw)

    # Exportar dataset limpio (para exploración y auditoría)
    clean_path = PROCESSED_DIR / "dataset_limpio.csv"
    df_clean.to_csv(clean_path, index=False, encoding="utf-8")
    log.info(f"✓ Dataset limpio exportado → {clean_path.name} ({df_clean.shape})")

    # Paso 3: Preparación ML
    df_ml = preparar_para_ml(df_clean)

    # Exportar dataset ML-ready (solo columnas relevantes para el modelo)
    feature_cols_ml = (
        [f"{c}_enc" for c in CATEGORICAL_FEATURES + ["Franja_Horaria"]] +
        [f"{c}_scaled" for c in NUMERIC_FEATURES + ["Indice_Congestion", "Ratio_Ruido_Velocidad"]] +
        ["Es_Hora_Pico", "Es_Diurno", "Supera_Limite_Normativo",
         "Mes", "Trimestre", f"{TARGET_COLUMN}_encoded"]
    )
    df_ml[feature_cols_ml].to_csv(
        PROCESSED_DIR / "dataset_ml_ready.csv", index=False, encoding="utf-8"
    )
    log.info(f"✓ Dataset ML-ready exportado → dataset_ml_ready.csv ({df_ml[feature_cols_ml].shape})")

    # Paso 4: Estadísticas para el Dashboard
    generar_stats_dashboard(df_clean)

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  ✅ FASE 2 COMPLETADA SIN ERRORES                        ║")
    log.info(f"║  Artefactos en: data/processed/ ({len(list(PROCESSED_DIR.iterdir()))} archivos) ║")
    log.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()