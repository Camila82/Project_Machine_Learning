import os
import json
import joblib
import numpy as np
from pathlib import Path
from flask import Blueprint, render_template, request

prediction_bp = Blueprint('prediction', __name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
METRICS_PATH = BASE_DIR / 'data' / 'processed' / 'model_metrics.json'

@prediction_bp.route('/prediction-system', methods=['GET', 'POST'])
def predict():
    best_model_name = "Random Forest" # Default
    model_metrics = {}
    
    # 1. Leer cuál fue el mejor modelo
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            model_metrics = json.load(f)
            best_model_name = model_metrics.get('best_model', 'Random Forest')

    model_filename = best_model_name.replace(' ', '_').lower() + '.pkl'
    model_path = MODELS_DIR / model_filename
    
    prediction_result = None
    probabilities = None

    # 2. Procesar el formulario cuando el usuario haga clic en "Predict"
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            indice_movilidad = float(request.form['indice_movilidad'])
            flujo_vehicular = float(request.form['flujo_vehicular'])
            velocidad = float(request.form['velocidad'])
            saturacion = float(request.form['saturacion'])
            is_peak_hour = int(request.form['peak_hour'])
            calidad_sensor = float(request.form['calidad_sensor'])

            # Calcular el Congestion Index (Feature Engineering de la Fase 3)
            congestion_index = (saturacion * 0.6) + ((indice_movilidad / 100) * 0.4)

            # Preparar el vector de entrada exacto como se entrenó:
            # ["Indice_Movilidad", "Flujo_Vehicular_veh_h", "Velocidad_Promedio_kmh", "Congestion_Index", "Is_Peak_Hour", "Calidad_Señal_Sensor"]
            X_input = np.array([[indice_movilidad, flujo_vehicular, velocidad, congestion_index, is_peak_hour, calidad_sensor]])

            # Cargar modelo y predecir
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                pred_idx = model.predict(X_input)[0]
                
                # Las clases originales ordenadas
                classes = model_metrics['models'][best_model_name]['classes']
                prediction_result = classes[pred_idx]

                # Obtener probabilidades si el modelo lo permite
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_input)[0]
                    probabilities = {classes[i]: round(probs[i] * 100, 2) for i in range(len(classes))}
            else:
                prediction_result = "Error: Model file not found."

        except Exception as e:
            prediction_result = f"Input Error: {str(e)}"

    return render_template(
        'prediction_system.html',
        best_model=best_model_name,
        prediction=prediction_result,
        probabilities=probabilities
    )