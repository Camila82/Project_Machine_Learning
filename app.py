"""
App principal – Contaminación Acústica Urbana en Colombia.
Solo contiene la creación de la app Flask y las rutas.
"""
from flask import Flask, render_template

from config.settings import FLASK_PORT, FLASK_DEBUG
from services.data_loader import load_raw_data, clean_data, get_original_info, get_cleaned_info
from services.data_analyzer import get_descriptive_stats, get_target_distribution, get_city_distribution, get_preview_data
from utils.variables import get_selected_variables

app = Flask(__name__, template_folder='Templates')


@app.route('/')
def index():
    """Página principal: muestra el análisis completo del dataset."""
    df = load_raw_data()
    if df is None:
        return render_template('index.html', error=True)

    df_clean = clean_data(df)

    return render_template(
        'index.html',
        error=False,
        info_original=get_original_info(df),
        info_depurado=get_cleaned_info(df_clean),
        variables=get_selected_variables(df),
        stats=get_descriptive_stats(df),
        sample_data=get_preview_data(df),
        distribucion_objetivo=get_target_distribution(df),
        distribucion_ciudad=get_city_distribution(df),
    )


if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT, use_reloader=False)
