import json
import plotly
import plotly.graph_objs as go

# Paleta de colores consistente con nuestro CSS Premium
COLORS = {
    'primary': '#00f0ff',
    'secondary': '#4f46e5',
    'bg_transparent': 'rgba(0,0,0,0)',
    'text': '#e0e0e0',
    'grid': 'rgba(255,255,255,0.08)'
}

def get_base_layout(title):
    """Retorna el layout base de Plotly adaptado a nuestro tema oscuro."""
    return go.Layout(
        title={'text': title, 'font': {'color': COLORS['text'], 'size': 18}},
        plot_bgcolor=COLORS['bg_transparent'],
        paper_bgcolor=COLORS['bg_transparent'],
        font={'color': COLORS['text'], 'family': 'Inter, sans-serif'},
        xaxis={'gridcolor': COLORS['grid'], 'zerolinecolor': COLORS['grid']},
        yaxis={'gridcolor': COLORS['grid'], 'zerolinecolor': COLORS['grid']},
        margin={'l': 40, 'r': 20, 't': 50, 'b': 40}
    )

def create_target_distribution_chart(stats_general):
    """Gráfica de barras para la distribución de la variable objetivo (Nivel de Ruido)."""
    if not stats_general:
        return None
        
    dist = stats_general.get('classification_dist', {})
    labels = list(dist.keys())
    values = list(dist.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels, 
            y=values, 
            marker_color=[COLORS['primary'], COLORS['secondary'], '#10b981', '#ef4444']
        )
    ])
    fig.layout = get_base_layout("Normative Classification Distribution")
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_correlation_heatmap(stats_corr):
    """Mapa de calor para la matriz de correlaciones numéricas."""
    if not stats_corr:
        return None
        
    cols = stats_corr.get('columns', [])
    matrix = stats_corr.get('matrix', [])
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=cols,
        y=cols,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig.layout = get_base_layout("Feature Correlation Matrix")
    
    # --- CORRECCIÓN PARA EVITAR QUE SE CORTE ---
    fig.update_layout(
        xaxis=dict(
            tickangle=-45,       # Inclina los textos a 45 grados para que no choquen
            automargin=True,     # Calcula el margen inferior automáticamente
            gridcolor=COLORS['grid'], 
            zerolinecolor=COLORS['grid']
        ),
        yaxis=dict(
            automargin=True,     # Calcula el margen izquierdo automáticamente
            gridcolor=COLORS['grid'], 
            zerolinecolor=COLORS['grid']
        ),
        margin=dict(l=10, r=20, t=50, b=10) # Reducimos los márgenes fijos para que automargin haga su trabajo
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)