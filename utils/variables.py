"""
Definición de las variables seleccionadas para el estudio.
Contiene la metadata de variables predictoras, objetivo y descartadas.
"""


def get_selected_variables(df):
    """
    Genera el diccionario completo de variables seleccionadas
    con su justificación, a partir del DataFrame original.
    """
    return {
        'objetivo': _get_target_variable(df),
        'predictoras': _get_predictor_variables(df),
        'descartadas': _get_discarded_variables(),
    }


def _get_target_variable(df):
    """Metadata de la variable objetivo: Clasificacion_Normativa."""
    return {
        'nombre': 'Clasificacion_Normativa',
        'descripcion': 'Clasificación del nivel de ruido según la normativa colombiana (Bajo, Moderado, Alto, Crítico)',
        'tipo': 'Categórica',
        'valores_unicos': df['Clasificacion_Normativa'].unique().tolist(),
        'distribucion': df['Clasificacion_Normativa'].value_counts().to_dict(),
    }


def _get_predictor_variables(df):
    """Metadata de las 10 variables predictoras con justificación."""
    return [
        {
            'nombre': 'Ciudad',
            'descripcion': 'Ciudad de Colombia donde se realizó la medición',
            'tipo': 'Categórica',
            'justificacion': 'Diferentes ciudades tienen distintos niveles de urbanización y tráfico',
            'valores_unicos': len(df['Ciudad'].unique()),
        },
        {
            'nombre': 'Hora',
            'descripcion': 'Hora del día en que se registró la medición (0-23)',
            'tipo': 'Numérica',
            'justificacion': 'El ruido varía significativamente según la hora del día',
            'rango': f"{int(df['Hora'].min())} - {int(df['Hora'].max())}",
        },
        {
            'nombre': 'Tipo_Zona',
            'descripcion': 'Tipo de zona urbana (Residencial, Comercial, Industrial, etc.)',
            'tipo': 'Categórica',
            'justificacion': 'El tipo de zona determina las fuentes de ruido predominantes',
            'valores_unicos': len(df['Tipo_Zona'].unique()),
        },
        {
            'nombre': 'Fuente_Principal_Ruido',
            'descripcion': 'Principal fuente generadora de ruido',
            'tipo': 'Categórica',
            'justificacion': 'La fuente de ruido es un predictor directo del nivel sonoro',
            'valores_unicos': len(df['Fuente_Principal_Ruido'].unique()),
        },
        {
            'nombre': 'Nivel_Ruido_dB',
            'descripcion': 'Nivel de ruido registrado en decibeles',
            'tipo': 'Numérica',
            'justificacion': 'Medición directa del ruido, fuertemente correlacionada con la clasificación',
            'rango': f"{df['Nivel_Ruido_dB'].min()} - {df['Nivel_Ruido_dB'].max()} dB",
        },
        {
            'nombre': 'Flujo_Vehicular_veh_h',
            'descripcion': 'Flujo vehicular en vehículos por hora',
            'tipo': 'Numérica',
            'justificacion': 'El tráfico vehicular es una de las principales fuentes de contaminación acústica',
            'rango': f"{int(df['Flujo_Vehicular_veh_h'].min())} - {int(df['Flujo_Vehicular_veh_h'].max())} veh/h",
        },
        {
            'nombre': 'Velocidad_Promedio_kmh',
            'descripcion': 'Velocidad promedio del tráfico en km/h',
            'tipo': 'Numérica',
            'justificacion': 'La velocidad del tráfico influye en la intensidad del ruido generado',
            'rango': f"{df['Velocidad_Promedio_kmh'].min()} - {df['Velocidad_Promedio_kmh'].max()} km/h",
        },
        {
            'nombre': 'Saturacion_Transporte',
            'descripcion': 'Índice de saturación del transporte (0.0 - 1.0)',
            'tipo': 'Numérica',
            'justificacion': 'Mayor saturación implica más congestión y por ende mayor ruido',
            'rango': f"{df['Saturacion_Transporte'].min()} - {df['Saturacion_Transporte'].max()}",
        },
        {
            'nombre': 'Condicion_Climatica',
            'descripcion': 'Condición climática al momento de la medición',
            'tipo': 'Categórica',
            'justificacion': 'Las condiciones climáticas afectan la propagación del sonido',
            'valores_unicos': len(df['Condicion_Climatica'].unique()),
        },
        {
            'nombre': 'Indice_Movilidad',
            'descripcion': 'Índice de movilidad urbana',
            'tipo': 'Numérica',
            'justificacion': 'Refleja la actividad de transporte en la zona',
            'rango': f"{int(df['Indice_Movilidad'].min())} - {int(df['Indice_Movilidad'].max())}",
        },
    ]


def _get_discarded_variables():
    """Variables descartadas con la razón de su exclusión."""
    return [
        {'nombre': 'ID_Registro', 'razon': 'Identificador único sin valor predictivo'},
        {'nombre': 'Fecha', 'razon': 'Alta cardinalidad (720 valores únicos) que no aporta un patrón claro sin ingeniería de features'},
        {'nombre': 'Latitud', 'razon': 'Coordenada geográfica ya representada por la variable Ciudad'},
        {'nombre': 'Longitud', 'razon': 'Coordenada geográfica ya representada por la variable Ciudad'},
        {'nombre': 'Observaciones', 'razon': 'Texto libre no estructurado, no apto para modelado directo'},
        {'nombre': 'Tipo_Sensor', 'razon': 'Artefacto de medición, no un factor que influya en el ruido real'},
        {'nombre': 'Calidad_Señal_Sensor', 'razon': 'Métrica de calidad del sensor, no un factor ambiental del ruido'},
    ]
