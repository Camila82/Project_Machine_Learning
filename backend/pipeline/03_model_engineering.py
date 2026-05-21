"""
==============================================================================
CRISP-ML | PHASE 3: MODEL ENGINEERING
Project : Urban Noise Analysis System — Colombian Cities
Version : 1.0.0
==============================================================================

Responsibilities:
  1. Load ML-ready dataset produced by Phase 2
  2. Train and compare four candidate algorithms
  3. Cross-validated hyperparameter tuning on the best candidate
  4. Full evaluation suite (accuracy, F1, confusion matrix, feature importance)
  5. Serialize winning model + scaler artifacts for the prediction API

Candidate models:
  - Logistic Regression     → interpretable baseline
  - K-Nearest Neighbors     → non-parametric reference
  - Random Forest           → primary candidate (ensemble, handles mixed types)
  - Gradient Boosting       → boosted candidate (sequential error correction)

Target: Clasificacion_Normativa_encoded  (0=Low · 1=Moderate · 2=High · 3=Critical)

Outputs written to data/models/:
  - best_model.pkl              → serialized winning classifier
  - model_comparison.json       → accuracy / F1 table for all four models
  - model_metrics.json          → full metrics of the winning model
  - feature_importance.json     → ranked feature importances
  - confusion_matrix.json       → confusion matrix (test set)
==============================================================================
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parents[2]
PROCESSED_DIR  = ROOT_DIR / "data" / "processed"
MODELS_DIR     = ROOT_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ML_READY_PATH  = PROCESSED_DIR / "dataset_ml_ready.csv"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TARGET          = "Clasificacion_Normativa_encoded"
TARGET_LABELS   = ["Low", "Moderate", "High", "Critical"]  # ordinal order
RANDOM_STATE    = 42
TEST_SIZE       = 0.20   # 80 / 20 split
CV_FOLDS        = 5      # stratified k-fold


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(path: Path):
    """Load ML-ready dataset and produce stratified train / test sets."""
    log.info("=" * 60)
    log.info("STEP 1: LOADING DATASET AND SPLITTING")
    log.info("=" * 60)

    df = pd.read_csv(path)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # preserve class proportions in both sets
    )

    log.info(f"Total samples  : {len(df)}")
    log.info(f"Features       : {X.shape[1]}")
    log.info(f"Train set      : {len(X_train)} samples ({100*(1-TEST_SIZE):.0f}%)")
    log.info(f"Test set       : {len(X_test)}  samples ({100*TEST_SIZE:.0f}%)")
    log.info(f"Class balance (train): {dict(y_train.value_counts().sort_index())}")

    return X_train, X_test, y_train, y_test, list(X.columns)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CANDIDATE MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_candidates(X_train, y_train) -> dict:
    """
    Train four candidate models with default parameters.
    Evaluate each using stratified 5-fold CV (accuracy + macro F1).
    Returns a dict of results sorted by mean CV accuracy.
    """
    log.info("=" * 60)
    log.info("STEP 2: CANDIDATE MODEL COMPARISON  (5-fold CV)")
    log.info("=" * 60)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }

    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results     = {}

    for name, model in candidates.items():
        acc_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="accuracy",       n_jobs=-1)
        f1_scores  = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="f1_macro",       n_jobs=-1)

        results[name] = {
            "cv_accuracy_mean":  round(float(acc_scores.mean()), 4),
            "cv_accuracy_std":   round(float(acc_scores.std()),  4),
            "cv_f1_macro_mean":  round(float(f1_scores.mean()),  4),
            "cv_f1_macro_std":   round(float(f1_scores.std()),   4),
        }

        log.info(
            f"  {name:<25} | Acc: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}"
            f" | F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}"
        )

    # Sort by CV accuracy descending
    results = dict(sorted(results.items(), key=lambda x: x[1]["cv_accuracy_mean"], reverse=True))
    best_name = next(iter(results))
    log.info(f"\n  → Best candidate: [{best_name}]  (CV Acc = {results[best_name]['cv_accuracy_mean']:.4f})")

    return results, best_name


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — HYPERPARAMETER TUNING (Grid Search on winning model)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth":    [None, 15],
        "max_features": ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators":  [100, 150],
        "learning_rate": [0.05, 0.10],
        "max_depth":     [3, 5],
    },
    "Logistic Regression": {
        "C":      [0.1, 1, 10],
        "solver": ["lbfgs"],
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [5, 7, 11],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan"],
    },
}

MODEL_CONSTRUCTORS = {
    "Random Forest":      lambda: RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting":  lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Logistic Regression":lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "K-Nearest Neighbors":lambda: KNeighborsClassifier(),
}


def tune_hyperparameters(best_name: str, X_train, y_train):
    """
    Run GridSearchCV on the winning candidate.
    Uses stratified 5-fold CV and optimizes for macro F1
    (more robust than accuracy on slightly imbalanced classes).
    Returns the fitted best estimator.
    """
    log.info("=" * 60)
    log.info(f"STEP 3: HYPERPARAMETER TUNING — {best_name}")
    log.info("=" * 60)

    base_model  = MODEL_CONSTRUCTORS[best_name]()
    param_grid  = PARAM_GRIDS[best_name]
    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        scoring    = "f1_macro",
        cv         = cv_strategy,
        n_jobs     = -1,
        verbose    = 0,
        refit      = True,       # refit on full training set with best params
    )

    log.info(f"  Grid combinations: {_count_grid(param_grid)} × {CV_FOLDS} folds")
    log.info("  Running grid search — this may take a moment...")
    grid_search.fit(X_train, y_train)

    log.info(f"  Best params  : {grid_search.best_params_}")
    log.info(f"  Best CV F1   : {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def _count_grid(param_grid: dict) -> int:
    total = 1
    for v in param_grid.values():
        total *= len(v)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FINAL EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_final_model(model, X_test, y_test, feature_names: list, best_name: str) -> dict:
    """
    Evaluate the tuned model on the held-out test set.
    Computes accuracy, per-class F1, confusion matrix, and feature importances.
    """
    log.info("=" * 60)
    log.info("STEP 4: FINAL EVALUATION ON HELD-OUT TEST SET")
    log.info("=" * 60)

    y_pred = model.predict(X_test)

    acc       = accuracy_score(y_test, y_pred)
    f1_macro  = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    report    = classification_report(
        y_test, y_pred,
        target_names=TARGET_LABELS,
        output_dict=True,
    )
    cm        = confusion_matrix(y_test, y_pred).tolist()

    log.info(f"  Test Accuracy     : {acc:.4f}  ({acc*100:.2f}%)")
    log.info(f"  Test F1 (macro)   : {f1_macro:.4f}")
    log.info(f"  Test F1 (weighted): {f1_weighted:.4f}")
    log.info(f"  Confusion matrix  :\n{confusion_matrix(y_test, y_pred)}")
    log.info("\n" + classification_report(y_test, y_pred, target_names=TARGET_LABELS))

    # ── Feature importances (tree-based models) ───────────────────────────────
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_pairs = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )
        feature_importance = {name: round(float(val), 6) for name, val in importance_pairs}
        log.info("  Top-5 features by importance:")
        for fname, fval in list(importance_pairs)[:5]:
            log.info(f"    {fname:<40} {fval:.4f}")
    elif hasattr(model, "coef_"):
        # Logistic Regression: use mean absolute coefficient across classes
        mean_coef = np.abs(model.coef_).mean(axis=0)
        importance_pairs = sorted(
            zip(feature_names, mean_coef), key=lambda x: x[1], reverse=True
        )
        feature_importance = {name: round(float(val), 6) for name, val in importance_pairs}

    # ── Build metrics dict ────────────────────────────────────────────────────
    per_class = {}
    for label in TARGET_LABELS:
        per_class[label] = {
            "precision": round(report[label]["precision"], 4),
            "recall":    round(report[label]["recall"],    4),
            "f1_score":  round(report[label]["f1-score"],  4),
            "support":   int(report[label]["support"]),
        }

    metrics = {
        "model_name":        best_name,
        "test_accuracy":     round(float(acc), 4),
        "test_f1_macro":     round(float(f1_macro), 4),
        "test_f1_weighted":  round(float(f1_weighted), 4),
        "per_class_metrics": per_class,
        "target_labels":     TARGET_LABELS,
        "n_test_samples":    int(len(y_test)),
        "n_train_features":  int(len(feature_names)),
    }

    return metrics, cm, feature_importance


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SERIALIZE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

def serialize_artifacts(
    model,
    best_name: str,
    best_params: dict,
    best_cv_f1: float,
    comparison: dict,
    metrics: dict,
    confusion_mat: list,
    feature_importance: dict,
    feature_names: list,
) -> None:
    """Persist all model artifacts to data/models/."""
    log.info("=" * 60)
    log.info("STEP 5: SERIALIZING ARTIFACTS")
    log.info("=" * 60)

    # ── 5.1 Model binary ─────────────────────────────────────────────────────
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(model, model_path)
    log.info(f"  ✓ Model saved       → {model_path.name}")

    # ── 5.2 Model comparison table ────────────────────────────────────────────
    _save_json({"candidates": comparison}, "model_comparison.json")

    # ── 5.3 Full metrics of winning model ─────────────────────────────────────
    metrics["best_params"]  = best_params
    metrics["best_cv_f1"]   = round(float(best_cv_f1), 4)
    metrics["feature_names"] = feature_names
    _save_json(metrics, "model_metrics.json")

    # ── 5.4 Confusion matrix ──────────────────────────────────────────────────
    _save_json({
        "matrix":        confusion_mat,
        "target_labels": TARGET_LABELS,
    }, "confusion_matrix.json")

    # ── 5.5 Feature importances ───────────────────────────────────────────────
    _save_json({
        "importances":  feature_importance,
        "feature_names": list(feature_importance.keys()),
        "values":        list(feature_importance.values()),
    }, "feature_importance.json")

    log.info(f"  ✓ All JSON artifacts → data/models/ ({len(list(MODELS_DIR.iterdir()))} files)")


def _save_json(data: dict, filename: str) -> None:
    path = MODELS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"  ✓ Saved             → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  CRISP-ML | PHASE 3 — MODEL ENGINEERING                 ║")
    log.info("║  Urban Noise Analysis — Colombia                        ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    # Step 1 — load & split
    X_train, X_test, y_train, y_test, feature_names = load_and_split(ML_READY_PATH)

    # Step 2 — compare candidates (CV)
    comparison, best_name = compare_candidates(X_train, y_train)

    # Step 3 — tune the winner
    best_model, best_params, best_cv_f1 = tune_hyperparameters(best_name, X_train, y_train)

    # Step 4 — final test-set evaluation
    metrics, confusion_mat, feature_importance = evaluate_final_model(
        best_model, X_test, y_test, feature_names, best_name
    )

    # Step 5 — persist all artifacts
    serialize_artifacts(
        model              = best_model,
        best_name          = best_name,
        best_params        = best_params,
        best_cv_f1         = best_cv_f1,
        comparison         = comparison,
        metrics            = metrics,
        confusion_mat      = confusion_mat,
        feature_importance = feature_importance,
        feature_names      = feature_names,
    )

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info(f"║  ✅ PHASE 3 COMPLETE                                    ║")
    log.info(f"║  Winner : {best_name:<44}  ║")
    log.info(f"║  Test Accuracy : {metrics['test_accuracy']:.4f}  |  F1 Macro : {metrics['test_f1_macro']:.4f}          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()