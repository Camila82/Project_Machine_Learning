from flask import Blueprint, render_template
from utils.data_loader import get_all_data_artifacts
from utils.charts import create_target_distribution_chart, create_correlation_heatmap

data_bp = Blueprint('data', __name__)

@data_bp.route('/data-understanding')
def understanding():
    # 1. Cargar datos
    artifacts = get_all_data_artifacts()
    
    # 2. Generar gráficas
    dist_chart = create_target_distribution_chart(artifacts['general'])
    corr_chart = create_correlation_heatmap(artifacts['correlations'])
    
    # 3. Renderizar plantilla pasando los datos y gráficas
    return render_template(
        'data_understanding.html',
        general=artifacts['general'],
        eda=artifacts['eda'],
        sample_data=artifacts['sample'],
        dist_chart=dist_chart,
        corr_chart=corr_chart
    )

@data_bp.route('/data-engineering')
def engineering():
    return render_template('data_engineering.html')