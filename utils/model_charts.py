import json
import plotly
import plotly.graph_objs as go
from utils.charts import COLORS, get_base_layout

def create_metrics_comparison_chart(metrics_data):
    """Bar chart comparing Accuracy, Precision, Recall, and F1 across models."""
    if not metrics_data or 'models' not in metrics_data:
        return None

    models = list(metrics_data['models'].keys())
    accuracy = [metrics_data['models'][m]['accuracy'] for m in models]
    f1_score = [metrics_data['models'][m]['f1_score'] for m in models]
    roc_auc = [metrics_data['models'][m].get('roc_auc', 0) for m in models]

    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=models, y=accuracy, marker_color=COLORS['primary']),
        go.Bar(name='F1-Score', x=models, y=f1_score, marker_color=COLORS['secondary']),
        go.Bar(name='ROC AUC', x=models, y=roc_auc, marker_color='#10b981')
    ])

    fig.layout = get_base_layout("Comparative Performance Analysis")
    fig.update_layout(barmode='group', yaxis_title='Score (%)', xaxis_title='Models')
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_confusion_matrix_charts(metrics_data):
    """Generates a list of Heatmap figures for each model's Confusion Matrix."""
    if not metrics_data or 'models' not in metrics_data:
        return {}

    cm_charts = {}
    for model_name, data in metrics_data['models'].items():
        z = data['confusion_matrix']
        classes = data['classes']
        
        fig = go.Figure(data=go.Heatmap(
            z=z, x=classes, y=classes,
            colorscale='Blues',
            text=z, texttemplate="%{text}",
            hoverongaps=False
        ))
        
        fig.layout = get_base_layout(f"{model_name} Matrix")
        fig.update_layout(
            xaxis_title="Predicted Label", 
            yaxis_title="True Label",
            margin=dict(l=50, r=20, t=50, b=50)
        )
        
        cm_charts[model_name] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    return cm_charts