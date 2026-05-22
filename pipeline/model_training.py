import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Rutas
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "dataset_ruido_urbano_colombia.csv"
MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data():
    """Reproduce características básicas y prepara X, y."""
    print("Loading and preparing data...")
    df = pd.read_csv(RAW_PATH, sep=';', encoding='latin-1', decimal=',')
    
    # Feature Engineering (Simplificado)
    df["Congestion_Index"] = (df["Saturacion_Transporte"] * 0.6 + (df["Indice_Movilidad"] / 100) * 0.4)
    peak_hours = list(range(7, 10)) + list(range(17, 20))
    df["Is_Peak_Hour"] = df["Hora"].isin(peak_hours).astype(int)
    
    features = ["Indice_Movilidad", "Flujo_Vehicular_veh_h", "Velocidad_Promedio_kmh", 
                "Congestion_Index", "Is_Peak_Hour", "Calidad_Señal_Sensor"]
    target = "Clasificacion_Normativa"
    
    # Limpiar nulos para estar seguros
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    # Codificar Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Escalar X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, features, le, scaler

def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, target_names):
    """Entrena y extrae métricas rigurosas."""
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Prediction Sample (Para la consola web)
    sample_preds = []
    for i in range(3): # Tomamos 3 predicciones de ejemplo
        sample_preds.append({
            "real": target_names[y_test[i]],
            "predicted": target_names[y_pred[i]],
            "status": "CORRECT" if y_test[i] == y_pred[i] else "INCORRECT"
        })

    return {
        "name": name,
        "accuracy": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": cm,
        "classes": list(target_names),
        "console_predictions": sample_preds
    }

def main():
    X_train, X_test, y_train, y_test, feature_names, le, scaler = prepare_data()
    target_names = le.classes_
    
    # 1. Definición de Modelos e Hiperparámetros
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }
    
    results = {}
    best_model = None
    best_f1 = 0
    
    # 2. Entrenamiento y Evaluación
    for name, clf in models.items():
        metrics = train_and_evaluate(name, clf, X_train, X_test, y_train, y_test, target_names)
        
        # Hyperparameters extraction
        metrics["hyperparameters"] = clf.get_params()
        
        # Feature Importance (Si aplica)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            metrics["feature_importance"] = {feature_names[i]: round(importances[i], 4) for i in range(len(feature_names))}
            
        results[name] = metrics
        
        # Guardar modelo físico
        joblib.dump(clf, MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl")
        
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = name
            
    # Guardar métricas globales
    final_report = {
        "dataset_split": {"train_size": len(X_train), "test_size": len(X_test)},
        "best_model": best_model,
        "models": results
    }
    
    with open(PROCESSED_DIR / "model_metrics.json", "w") as f:
        json.dump(final_report, f, indent=4, default=str)
        
    print(f"✅ Training complete. Best model: {best_model} (F1: {best_f1}%)")

if __name__ == "__main__":
    main()