import json
import plotly
import plotly.graph_objs as go
from utils.charts import COLORS, get_base_layout

def create_reward_progression_chart(smoothed_rewards):
    """Line chart showing how the agent maximizes rewards over time."""
    fig = go.Figure()
    
    # Eje X representará los bloques de episodios
    x_axis = [i * 50 for i in range(len(smoothed_rewards))]
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=smoothed_rewards,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=3, shape='spline'),
        marker=dict(size=6, color=COLORS['secondary'], line=dict(width=1, color='white')),
        fill='tozeroy',
        fillcolor='rgba(0, 240, 255, 0.1)'
    ))
    
    fig.layout = get_base_layout("Agent Learning Curve (Reward Progression)")
    fig.update_layout(
        xaxis_title="Training Episodes", 
        yaxis_title="Average Cumulative Reward",
        margin=dict(l=50, r=20, t=50, b=50)
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_qtable_heatmap(q_table):
    """Heatmap visualization of the learned Policy (Q-Table)."""
    states = ['Low Noise', 'Moderate Noise', 'High Noise', 'Critical Noise']
    actions = ['Keep Lights', 'Switch Lights']
    
    # Redondear valores para la gráfica
    rounded_q = [[round(val, 2) for val in row] for row in q_table]

    fig = go.Figure(data=go.Heatmap(
        z=rounded_q,
        x=actions,
        y=states,
        colorscale='Plasma',
        text=rounded_q,
        texttemplate="%{text}",
        hoverongaps=False
    ))
    
    fig.layout = get_base_layout("Learned Policy Matrix (Q-Table)")
    fig.update_layout(margin=dict(l=100, r=20, t=50, b=50))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)