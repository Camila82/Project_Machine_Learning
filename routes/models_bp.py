import json
import os
from pathlib import Path
from flask import Blueprint, render_template

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