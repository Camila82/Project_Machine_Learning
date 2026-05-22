from flask import Blueprint, render_template
from utils.q_learning import train_smart_traffic_agent
from utils.rl_charts import create_reward_progression_chart, create_qtable_heatmap

rl_bp = Blueprint('rl', __name__)

@rl_bp.route('/rl-concepts')
def concepts():
    # Renderiza la vista teórica (La construiremos en el siguiente paso)
    return render_template('rl_concepts.html')

@rl_bp.route('/rl-application')
def application():
    # 1. Ejecutar el entrenamiento del agente (1000 episodios)
    agent_results = train_smart_traffic_agent()
    
    # 2. Generar gráficas con los resultados del entrenamiento
    reward_chart = create_reward_progression_chart(agent_results['smoothed_rewards'])
    q_table_chart = create_qtable_heatmap(agent_results['q_table'])
    
    # 3. Pasar resultados y gráficas a la plantilla web
    return render_template(
        'rl_application.html',
        final_score=agent_results['final_performance'],
        reward_chart=reward_chart,
        q_table_chart=q_table_chart
    )