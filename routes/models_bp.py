import json
import os
from pathlib import Path
from flask import Blueprint, render_template
from utils.model_charts import create_metrics_comparison_chart, create_confusion_matrix_charts

models_bp = Blueprint('models', __name__)

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / 'data' / 'processed' / 'model_metrics.json'

@models_bp.route('/model-engineering')
def engineering():
    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
            
    return render_template('model_engineering.html', metrics=metrics_data)

@models_bp.route('/model-evaluation')
def evaluation():
    metrics_data = {}
    comparison_chart = None
    cm_charts = {}
    
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
            
        comparison_chart = create_metrics_comparison_chart(metrics_data)
        cm_charts = create_confusion_matrix_charts(metrics_data)
            
    return render_template(
        'model_evaluation.html', 
        metrics=metrics_data,
        comparison_chart=comparison_chart,
        cm_charts=cm_charts
    )