"""
=============================================================================
URBAN NOISE ANALYSIS — COLOMBIAN CITIES
Pipeline: Data Engineering
CRISP-ML Phase 2 & 3: Data Understanding + Data Engineering

Run this script ONCE locally before starting the Flask application.
It reads the raw dataset, performs all cleaning and transformations,
and writes JSON artifacts to data/processed/.

Flask never runs this script — it only reads its outputs.

Outputs written to data/processed/:
  - stats_general.json       Global KPIs for the dashboard home cards
  - stats_by_city.json       Per-city aggregated metrics
  - stats_by_zone.json       Per-zone metrics + normative limits
  - stats_by_hour.json       Hourly noise distribution (0–23 h)
  - stats_correlations.json  Numeric feature correlation matrix
  - stats_eda.json           Variable catalog: types, nulls, ranges, uniques
  - dataset_sample.json      First 50 clean rows for the data table display
  - encoding_map.json        Encoding metadata (used in Phase 3 predictions)
=============================================================================
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — always relative to THIS file so they work on any machine
# ---------------------------------------------------------------------------
ROOT_DIR      = Path(__file__).resolve().parent.parent
RAW_PATH      = ROOT_DIR / "data" / "raw" / "dataset_ruido_urbano_colombia.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Colombian noise normative limits in dB — Resolución 627/2006 MADS
NORMATIVE_LIMITS = {
    "Residencial":  {"day": 65, "night": 55},
    "Comercial":    {"day": 70, "night": 60},
    "Industrial":   {"day": 75, "night": 70},
    "Hospitalaria": {"day": 55, "night": 45},
    "Educativa":    {"day": 65, "night": 55},
    "Mixta":        {"day": 68, "night": 58},
}

# Time bands based on Colombian regulation
TIME_BANDS = {
    "Early Morning": range(0, 6),
    "Morning":       range(6, 12),
    "Afternoon":     range(12, 18),
    "Night":         range(18, 24),
}

# Ordinal target mapping
TARGET_ORDER  = ["Bajo", "Moderado", "Alto", "Crítico"]
TARGET_LABELS = ["Low",  "Moderate", "High", "Critical"]

# Numeric features used for ML (Phase 3)
NUMERIC_FEATURES = [
    "Nivel_Ruido_dB",
    "Indice_Movilidad",
    "Flujo_Vehicular_veh_h",
    "Velocidad_Promedio_kmh",
    "Saturacion_Transporte",
    "Calidad_Señal_Sensor",
]

CATEGORICAL_FEATURES = [
    "Ciudad",
    "Tipo_Zona",
    "Tipo_Sensor",
    "Fuente_Principal_Ruido",
    "Condicion_Climatica",
]


# ===========================================================================
# STEP 1 — Load and validate raw data
# ===========================================================================

def load_raw(path: Path) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 1 — Loading and validating raw dataset")
    log.info("=" * 60)

    if not path.exists():
        log.error(f"Dataset not found: {path}")
        log.error("Place the CSV file at: data/raw/dataset_ruido_urbano_colombia.csv")
        sys.exit(1)

    df = pd.read_csv(path, sep=';', parse_dates=["Fecha"], encoding='latin-1', decimal=',')
    log.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Null audit
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls == 0:
        log.info("Null values: 0 (dataset is complete)")
    else:
        log.warning(f"Null values found:\n{null_counts[null_counts > 0]}")

    # Duplicate audit
    dupes = df.duplicated(subset=["ID_Registro"]).sum()
    log.info(f"Duplicate IDs: {dupes}")

    # Range check on main acoustic variable
    log.info(
        f"Nivel_Ruido_dB → "
        f"min={df['Nivel_Ruido_dB'].min():.1f} | "
        f"max={df['Nivel_Ruido_dB'].max():.1f} | "
        f"mean={df['Nivel_Ruido_dB'].mean():.2f}"
    )
    log.info(
        f"Target distribution:\n"
        f"{df['Clasificacion_Normativa'].value_counts().to_string()}"
    )
    return df


# ===========================================================================
# STEP 2 — Feature engineering
# ===========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 2 — Feature engineering")
    log.info("=" * 60)

    df = df.copy()

    # Feature 1: Time band (based on Colombian regulation)
    def assign_band(hour: int) -> str:
        for band, rng in TIME_BANDS.items():
            if hour in rng:
                return band
        return "Night"

    df["Time_Band"] = df["Hora"].apply(assign_band)
    log.info(f"Time_Band: {df['Time_Band'].value_counts().to_dict()}")

    # Feature 2: Peak hour flag (7–9 h and 17–19 h, Colombian traffic pattern)
    peak_hours = list(range(7, 10)) + list(range(17, 20))
    df["Is_Peak_Hour"] = df["Hora"].isin(peak_hours).astype(int)
    log.info(f"Is_Peak_Hour: {df['Is_Peak_Hour'].sum()} peak measurements")

    # Feature 3: Daytime flag (6–21 h)
    df["Is_Daytime"] = df["Hora"].between(6, 21).astype(int)

    # Feature 4: Exceeds normative limit (Resolution 627/2006)
    def exceeds_limit(row) -> int:
        zone = row["Tipo_Zona"]
        db   = row["Nivel_Ruido_dB"]
        band = row["Time_Band"]
        if zone not in NORMATIVE_LIMITS:
            return 0
        is_night = band in ("Early Morning", "Night")
        limit = NORMATIVE_LIMITS[zone]["night" if is_night else "day"]
        return int(db > limit)

    df["Exceeds_Limit"] = df.apply(exceeds_limit, axis=1)
    pct = df["Exceeds_Limit"].mean() * 100
    log.info(f"Exceeds_Limit: {df['Exceeds_Limit'].sum()} records ({pct:.1f}%)")

    # Feature 5: Congestion index (mobility + saturation combined)
    df["Congestion_Index"] = (
        df["Saturacion_Transporte"] * 0.6 +
        (df["Indice_Movilidad"] / 100) * 0.4
    ).round(4)

    # Feature 6: Noise-to-speed ratio
    df["Noise_Speed_Ratio"] = (
        df["Nivel_Ruido_dB"] /
        df["Velocidad_Promedio_kmh"].replace(0, np.nan)
    ).fillna(0).round(4)

    # Feature 7: Month and quarter for seasonality
    df["Month"]   = df["Fecha"].dt.month
    df["Quarter"] = df["Fecha"].dt.quarter

    log.info("Feature engineering complete — 7 new columns created")
    return df


# ===========================================================================
# STEP 3 — Generate all JSON artifacts
# ===========================================================================

def build_stats_general(df: pd.DataFrame) -> dict:
    return {
        "total_records":       int(len(df)),
        "cities_covered":      int(df["Ciudad"].nunique()),
        "date_range": {
            "start": str(df["Fecha"].min().date()),
            "end":   str(df["Fecha"].max().date()),
        },
        "noise_level": {
            "min":    round(float(df["Nivel_Ruido_dB"].min()), 1),
            "max":    round(float(df["Nivel_Ruido_dB"].max()), 1),
            "mean":   round(float(df["Nivel_Ruido_dB"].mean()), 2),
            "median": round(float(df["Nivel_Ruido_dB"].median()), 2),
            "std":    round(float(df["Nivel_Ruido_dB"].std()), 2),
        },
        "pct_exceeds_normative": round(float(df["Exceeds_Limit"].mean() * 100), 2),
        "pct_peak_hour":         round(float(df["Is_Peak_Hour"].mean() * 100), 2),
        "classification_dist": {
            k: int(v) for k, v in
            df["Clasificacion_Normativa"].value_counts().items()
        },
        "noise_source_dist": {
            k: int(v) for k, v in
            df["Fuente_Principal_Ruido"].value_counts().items()
        },
        "sensor_type_dist": {
            k: int(v) for k, v in
            df["Tipo_Sensor"].value_counts().items()
        },
        "zone_type_dist": {
            k: int(v) for k, v in
            df["Tipo_Zona"].value_counts().items()
        },
    }


def build_stats_by_city(df: pd.DataFrame) -> dict:
    result = {}
    for city, grp in df.groupby("Ciudad"):
        result[city] = {
            "total_measurements":    int(len(grp)),
            "avg_noise_db":          round(float(grp["Nivel_Ruido_dB"].mean()), 2),
            "max_noise_db":          round(float(grp["Nivel_Ruido_dB"].max()), 1),
            "min_noise_db":          round(float(grp["Nivel_Ruido_dB"].min()), 1),
            "pct_critical":          round(float((grp["Clasificacion_Normativa"] == "Crítico").mean() * 100), 2),
            "pct_exceeds_normative": round(float(grp["Exceeds_Limit"].mean() * 100), 2),
            "avg_mobility_index":    round(float(grp["Indice_Movilidad"].mean()), 2),
            "avg_vehicle_flow":      round(float(grp["Flujo_Vehicular_veh_h"].mean()), 1),
            "classification_dist":   {k: int(v) for k, v in grp["Clasificacion_Normativa"].value_counts().items()},
            "noise_source_dist":     {k: int(v) for k, v in grp["Fuente_Principal_Ruido"].value_counts().items()},
            "lat_center":            round(float(grp["Latitud"].mean()), 4),
            "lon_center":            round(float(grp["Longitud"].mean()), 4),
        }
    return result


def build_stats_by_zone(df: pd.DataFrame) -> dict:
    result = {}
    for zone, grp in df.groupby("Tipo_Zona"):
        limits = NORMATIVE_LIMITS.get(zone, {})
        result[zone] = {
            "total_measurements":    int(len(grp)),
            "avg_noise_db":          round(float(grp["Nivel_Ruido_dB"].mean()), 2),
            "max_noise_db":          round(float(grp["Nivel_Ruido_dB"].max()), 1),
            "pct_exceeds_normative": round(float(grp["Exceeds_Limit"].mean() * 100), 2),
            "day_limit_db":          limits.get("day", "N/A"),
            "night_limit_db":        limits.get("night", "N/A"),
            "classification_dist":   {k: int(v) for k, v in grp["Clasificacion_Normativa"].value_counts().items()},
        }
    return result


def build_stats_by_hour(df: pd.DataFrame) -> dict:
    result = {}
    for hour, grp in df.groupby("Hora"):
        result[str(int(hour))] = {
            "total_measurements": int(len(grp)),
            "avg_noise_db":       round(float(grp["Nivel_Ruido_dB"].mean()), 2),
            "max_noise_db":       round(float(grp["Nivel_Ruido_dB"].max()), 1),
            "time_band":          grp["Time_Band"].iloc[0],
            "pct_critical":       round(float((grp["Clasificacion_Normativa"] == "Crítico").mean() * 100), 2),
            "is_peak":            int(hour) in (list(range(7, 10)) + list(range(17, 20))),
        }
    return result


def build_stats_correlations(df: pd.DataFrame) -> dict:
    num_cols = NUMERIC_FEATURES + ["Congestion_Index", "Noise_Speed_Ratio",
                                    "Is_Peak_Hour", "Is_Daytime", "Exceeds_Limit"]
    corr = df[num_cols].corr().round(4)
    return {
        "columns": num_cols,
        "matrix":  corr.values.tolist(),
        "corr_with_noise": {
            k: round(float(v), 4) for k, v in
            corr["Nivel_Ruido_dB"].sort_values(ascending=False).items()
        },
    }


def build_stats_eda(df: pd.DataFrame) -> dict:
    variables = []
    for col in df.columns:
        if col in ("ID_Registro", "Observaciones"):
            continue  # skip non-analytical columns

        entry = {
            "name":       col,
            "null_count": int(df[col].isnull().sum()),
            "null_pct":   round(float(df[col].isnull().mean() * 100), 2),
            "unique":     int(df[col].nunique()),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            entry["type"] = "numeric"
            entry["stats"] = {
                "min":  round(float(df[col].min()), 4),
                "max":  round(float(df[col].max()), 4),
                "mean": round(float(df[col].mean()), 4),
                "std":  round(float(df[col].std()), 4),
                "q25":  round(float(df[col].quantile(0.25)), 4),
                "q50":  round(float(df[col].quantile(0.50)), 4),
                "q75":  round(float(df[col].quantile(0.75)), 4),
            }
        else:
            entry["type"] = "categorical"
            top = df[col].value_counts().head(5)
            entry["top_values"] = {k: int(v) for k, v in top.items()}

        variables.append(entry)

    return {
        "total_records":  int(len(df)),
        "total_columns":  int(len(df.columns)),
        "numeric_cols":   int(df.select_dtypes(include="number").shape[1]),
        "categorical_cols": int(df.select_dtypes(exclude="number").shape[1]),
        "total_nulls":    int(df.isnull().sum().sum()),
        "variables":      variables,
    }


def build_dataset_sample(df: pd.DataFrame) -> list:
    display_cols = [
        "ID_Registro", "Ciudad", "Fecha", "Hora", "Tipo_Zona",
        "Tipo_Sensor", "Fuente_Principal_Ruido", "Nivel_Ruido_dB",
        "Clasificacion_Normativa", "Indice_Movilidad",
        "Flujo_Vehicular_veh_h", "Velocidad_Promedio_kmh",
        "Condicion_Climatica",
    ]
    sample = df[display_cols].head(50).copy()
    sample["Fecha"] = sample["Fecha"].dt.strftime("%Y-%m-%d")
    return json.loads(sample.to_json(orient="records", force_ascii=False))


def build_encoding_map(df: pd.DataFrame) -> dict:
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler

    cat_encodings = {}
    for col in CATEGORICAL_FEATURES + ["Time_Band"]:
        le = LabelEncoder()
        le.fit(df[col])
        cat_encodings[col] = {
            str(cls): int(le.transform([cls])[0])
            for cls in le.classes_
        }

    scale_cols = NUMERIC_FEATURES + ["Congestion_Index", "Noise_Speed_Ratio"]
    scaler = MinMaxScaler()
    scaler.fit(df[scale_cols])

    return {
        "target_map":             {label: idx for idx, label in enumerate(TARGET_ORDER)},
        "target_order_es":        TARGET_ORDER,
        "target_order_en":        TARGET_LABELS,
        "categorical_encodings":  cat_encodings,
        "numeric_features_scaled": scale_cols,
        "scaler_min":             scaler.data_min_.tolist(),
        "scaler_max":             scaler.data_max_.tolist(),
    }


# ===========================================================================
# STEP 4 — Save artifacts
# ===========================================================================

def clean_json_keys(obj):
    """Convierte recursivamente todas las llaves de un diccionario a strings para evitar errores de JSON."""
    if isinstance(obj, dict):
        return {str(k): clean_json_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_keys(item) for item in obj]
    return obj

def save_json(data, filename: str) -> None:
    path = PROCESSED_DIR / filename
    data_clean = clean_json_keys(data) # Aplicamos la limpieza antes de guardar
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_clean, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"  Saved → {filename}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  Urban Noise Colombia — Data Engineering Pipeline       ║")
    log.info("║  CRISP-ML Phase 2 & 3                                   ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    # Step 1: Load
    df_raw = load_raw(RAW_PATH)

    # Step 2: Engineer features
    df = engineer_features(df_raw)

    # Step 3: Build all artifacts
    log.info("=" * 60)
    log.info("STEP 3 — Building and saving JSON artifacts")
    log.info("=" * 60)

    save_json(build_stats_general(df),      "stats_general.json")
    save_json(build_stats_by_city(df),      "stats_by_city.json")
    save_json(build_stats_by_zone(df),      "stats_by_zone.json")
    save_json(build_stats_by_hour(df),      "stats_by_hour.json")
    save_json(build_stats_correlations(df), "stats_correlations.json")
    save_json(build_stats_eda(df),          "stats_eda.json")
    save_json(build_dataset_sample(df),     "dataset_sample.json")
    save_json(build_encoding_map(df),       "encoding_map.json")

    log.info("=" * 60)
    artifacts = list(PROCESSED_DIR.glob("*.json"))
    log.info(f"Pipeline complete — {len(artifacts)} artifacts in data/processed/")
    for a in sorted(artifacts):
        size_kb = a.stat().st_size / 1024
        log.info(f"  {a.name:<35} {size_kb:.1f} KB")

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  ✅  All artifacts ready. You can now run Flask.        ║")
    log.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()