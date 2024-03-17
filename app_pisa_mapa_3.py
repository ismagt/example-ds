import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pickle
import numpy as np
import pandas as pd
import json
import dash_bootstrap_components as dbc
import shap
from sklearn.ensemble import RandomForestClassifier


with open('train_test_data.pkl', 'rb') as file:
    X_train, X_test, y_train, y_test = pickle.load(file)

geojson_path = 'map/spain-communities.geojson'
with open(geojson_path) as file:
    spain_geojson = json.load(file)
    
with open('df_vis.pkl', 'rb') as file:
    df_vis = pickle.load(file)


with open('shap_values.pkl', 'rb') as file:
    shap_values = pickle.load(file)

with open('feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)


# Supongamos que 'reducer' es un paso de preprocesamiento, como PCA, y tienes un modelo predictivo después
modelo_final = RandomForestClassifier()  # Por ejemplo, un clasificador

# Entrenar tu pipeline (suponiendo que X_train y y_train ya están definidos)
modelo_final.fit(X_test, y_test)


# Suponiendo que 'modelo_final' es tu modelo RandomForest o similar
explainer = shap.TreeExplainer(modelo_final)

# Calcular los valores SHAP (cambiar X_test por tus datos de prueba)
shap_values = explainer.shap_values(X_test)

# Ahora, X_train y X_test son DataFrames y puedes acceder a sus índices
test_indices = X_test.index


# Usando un CDN para Bootstrap
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
community_options = [{'label': 'All communities', 'value': 'All'}] + \
                    [{'label': community, 'value': community} for community in df_vis['Community'].unique()]

# Layout de la aplicación Dash
app.layout = html.Div([
    dbc.Container(fluid=True, children=[  # Contenedor principal
        dbc.Row([
            dbc.Col(html.Img(src='/assets/banner_2.png', style={'width': '100%', 'height': 'auto'}), width=12)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='3d-plot'), width=8),
            dbc.Col(
                # Envuelve el gráfico SHAP con dcc.Loading
                dcc.Loading(
                    id="loading-shap-bar-chart",
                    children=[dcc.Graph(id='shap-bar-chart')],
                    type="default",  # Puedes elegir otros estilos como "circle", "dot", etc.
                ), 
                width=4
            ),
        ]),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='community-dropdown', 
                                 options=community_options, 
                                 value='All', 
                                 clearable=False), width=8),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='histogram-repeat'), width=2),
            dbc.Col(dcc.Graph(id='histogram-level'), width=2),
            dbc.Col(dcc.Graph(id='histogram-gender'), width=2),
            dbc.Col(dcc.Graph(id='histogram-school-type'), width=2),
            dbc.Col(dcc.Graph(id='community-map'), width=4),
        ]),
    ]),
    html.Footer(
        [
            html.Div(
                [
                    html.P(
                        "Designed by Ismael Gómez-Talal",
                        className="footer-text"
                    ),
                    html.Img(
                        src=app.get_asset_url('logo_urjc.png'),
                        className="footer-image"
                    )
                ],
                className="footer-container"
            )
        ],
        className="footer"
    ) 
])

# Callback para la actualización de la visualización 3D y los histogramas
@app.callback(
    [
        Output('3d-plot', 'figure'),
        Output('histogram-repeat', 'figure'),
        Output('histogram-level', 'figure'),
        Output('histogram-gender', 'figure'),
        Output('histogram-school-type', 'figure'),
        Output('shap-bar-chart', 'figure'),
        Output('community-map', 'figure')
    ],
    [Input('community-dropdown', 'value'),
     Input('3d-plot', 'clickData')]
)

def update_output(selected_community, clickData):
    print(clickData)
    # Extraer el índice del punto seleccionado del clickData
    if clickData is not None:
        point_index = clickData['points'][0]['pointNumber']
    else:
        point_index = None  # O manejar de otra manera si no hay punto seleccionado

    # Filtrar el DataFrame basado en la comunidad seleccionada, o mostrar todas si 'All' es seleccionado
    if selected_community == 'All':
        filtered_df = df_vis
    else:
        filtered_df = df_vis[df_vis['Community'] == selected_community]
    
    # Inicializar figuras como figuras vacías o con un estado base
    fig_3d = go.Figure()  # o tu configuración inicial
    fig_repeat = go.Figure()
    fig_level = go.Figure()
    fig_gender = go.Figure()
    fig_school_type = go.Figure()
    fig_shap = go.Figure()

    # Define un mapa de colores para mantener los colores consistentes
    color_map = px.colors.qualitative.Plotly

    # Si seleccionas 'Todas', asegúrate de aplicar el mapa de colores
    if selected_community == 'All':
        for i, community in enumerate(df_vis['Community'].unique()):
            subset = df_vis[df_vis['Community'] == community]
            hover_text = subset.apply(lambda row: f"Autonomous_Community: {row['Community']}<br>" +
                                                f"REPEAT: {row['REPEAT']}<br>" +
                                                f"Gender: {row['Gender']}<br>" +
                                                f"School_Type: {row['School_Type']}<br>" +
                                                f"Level: {row['level']}", axis=1).tolist()
            fig_3d.add_trace(go.Scatter3d(
                # Tus parámetros...
                marker=dict(size=5, color=color_map[i % len(color_map)]),  # Aplica el color del mapa de colores
                x=subset['Component_1'],
                y=subset['Component_2'],
                z=subset['Component_3'],
                mode='markers',
                name=community,
                text=hover_text,
                hoverinfo='text'
            ))
    else:
            
        for community in filtered_df['Community'].unique():
            subset = df_vis[df_vis['Community'] == community]
            hover_text = subset.apply(lambda row: f"Autonomous_Community: {row['Community']}<br>" +
                                                f"REPEAT: {row['REPEAT']}<br>" +
                                                f"Gender: {row['Gender']}<br>" +
                                                f"School_Type: {row['School_Type']}<br>" +
                                                f"Level: {row['level']}", axis=1).tolist()

            fig_3d.add_trace(go.Scatter3d(
                x=subset['Component_1'],
                y=subset['Component_2'],
                z=subset['Component_3'],
                mode='markers',
                marker=dict(size=5),
                name=community,
                text=hover_text,
                hoverinfo='text'
            ))
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(xaxis_title='Componente 1', yaxis_title='Componente 2', zaxis_title='Componente 3'))
    
    # Replace this with your actual DataFrame
    data = {
        'Repeat Status': ['No Repeat', 'Repeat'],
        'Count': [filtered_df[filtered_df['REPEAT'] == 'No'].shape[0], 
                filtered_df[filtered_df['REPEAT'] == 'Yes'].shape[0]]
    }
    df_repeat = pd.DataFrame(data)

    fig_repeat = px.treemap(df_repeat, path=['Repeat Status'], values='Count', 
                    title='Repeat Status among Students')
    
    # The textinfo argument determines which information appears on the branches
    fig_repeat.update_traces(textinfo='label+percent parent')

    # Define your soft color palette
    soft_colors = ['lightgreen', 'red']  # Soft blue for 'No Repeat', Soft red for 'Repeat'

    # Update the colors of the treemap
    fig_repeat.update_traces(marker_colors=soft_colors)


    # Update the layout for a more pleasant aesthetic
    fig_repeat.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),  # Adjust the margins if necessary
    )


    # Sort the DataFrame by the 'level' so that 'Low' comes first, 'Medium' second, and 'High' last
    df_sorted = filtered_df.sort_values(by='level', key=lambda x: x.map({'Low': 1, 'Medium': 2, 'High': 3}))

    # Count the number of students at each level
    level_counts = df_sorted['level'].value_counts(normalize=True).mul(100).reindex(['Low', 'Medium', 'High']).fillna(0)

    # Create the gauge chart
    fig_level = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=level_counts['Medium'],
        delta={'reference': level_counts['Low'], 'increasing': {'color': "RebeccaPurple"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, level_counts['Low']], 'color': 'red'},
                {'range': [level_counts['Low'], level_counts['Low'] + level_counts['Medium']], 'color': 'yellow'},
                {'range': [level_counts['Low'] + level_counts['Medium'], 100], 'color': 'green'},
            ],
            'threshold': {
                'line': {'color': "RebeccaPurple", 'width': 4},
                'thickness': 0.75,
                'value': 46.8}
        }
    ))

    fig_level.update_layout(
        title='Student Level Distribution',
        font=dict(size=15, color='white'),  # Cambiar el color del texto a blanco para contraste
        paper_bgcolor='rgba(17,17,17,1)',  # Un gris oscuro como fondo del gráfico
        plot_bgcolor='rgba(17,17,17,1)',  # Un gris oscuro para el área del gráfico
        # Configuración de las anotaciones para que combinen con el diseño oscuro
        annotations=[
            dict(
                x=0.05,
                y=0,
                xref='paper',
                yref='paper',
                showarrow=False,
                text="Low",
                font=dict(size=15, color="red"),
                bgcolor='rgba(17,17,17,1)'  # Fondo gris oscuro para la anotación
            ),
            dict(
                x=0.5,
                y=0,
                xref='paper',
                yref='paper',
                showarrow=False,
                text="Medium",
                font=dict(size=15, color="yellow"),
                bgcolor='rgba(17,17,17,1)'  # Fondo gris oscuro para la anotación
            ),
            dict(
                x=0.95,
                y=0,
                xref='paper',
                yref='paper',
                showarrow=False,
                text="High",
                font=dict(size=15, color="green"),
                bgcolor='rgba(17,17,17,1)'  # Fondo gris oscuro para la anotación
            )
        ],
        # Configurar la leyenda para que sea visible en el fondo oscuro
        legend=dict(
            font=dict(color='white'),  # Color de texto de la leyenda
            bgcolor='rgba(17,17,17,1)',  # Fondo de la leyenda
            bordercolor='rgba(17,17,17,1)'  # Borde de la leyenda
        )
    )

    # Actualizar el indicador con un esquema de colores que funcione bien en fondos oscuros
    fig_level.update_traces(
        gauge=dict(
            axis=dict(
                tickcolor="white",  # Color de las marcas del eje
                tickfont=dict(color="white")  # Color del texto de las marcas
            ),
            bar=dict(color="darkgray")  # Color de la barra indicadora más oscuro para contraste
        )
    )

    # Replace this with your actual DataFrame
    data = {
        'Gender': ['Boy', 'Girl'],
        'Count': [filtered_df[filtered_df['Gender'] == 'Boy'].shape[0], 
                filtered_df[filtered_df['Gender'] == 'Girl'].shape[0]]
    }
    df_gender = pd.DataFrame(data)

    fig_gender = px.pie(df_gender, values='Count', names='Gender', title='Gender Distribution')

    # Customizing the pie chart colors
    colors = ['lightblue', 'pink']  # First color for Boys, second color for Girls

    fig_gender.update_traces(marker=dict(colors=colors))

    # Customizing the pie chart with annotations
    annotations = [
        dict(text='♀', x=0.20, y=0.35, font_size=35, showarrow=False),
        dict(text='♂', x=0.80, y=0.35, font_size=35, showarrow=False),
    ]

    fig_gender.update_layout(annotations=annotations)

    #fig_school_type = px.histogram(filtered_df, x='School_Type', title='School Type')

    # Contar los valores de 'School_Type'
    school_type_counts = filtered_df['School_Type'].value_counts().reset_index()
    school_type_counts.columns = ['School_Type', 'Count']

    # Calcular porcentajes
    school_type_counts['Percentage'] = (school_type_counts['Count'] / school_type_counts['Count'].sum()) * 100

    # Definir colores para las barras que contrasten bien con el fondo oscuro
    color_discrete_map = {'Public': '#EC407A', 'Private': '#29B6F6'}

    # Crear el gráfico de barras con el mapa de colores personalizado
    fig_school_type = px.bar(
        school_type_counts,
        x='School_Type',
        y='Percentage',
        title='School Type Distribution',
        text='Percentage',  # Esto agregará el porcentaje a las barras
        color='School_Type',  # Asegúrate de que este sea el nombre de tu columna
        color_discrete_map=color_discrete_map  # Usar el mapa de colores personalizado
    )

    # Actualizar las trazas para mostrar los porcentajes sobre las barras
    fig_school_type.update_traces(
        texttemplate='%{text:.2f}%',  # Formato de porcentaje con dos decimales
        textposition='inside'
    )

    # Actualizar el layout del gráfico para coincidir con el diseño oscuro
    fig_school_type.update_layout(
        xaxis_title="Type of School",
        yaxis_title="Percentage of Students",
        plot_bgcolor='rgba(17,17,17,1)',  # Fondo oscuro para el área del gráfico
        paper_bgcolor='rgba(17,17,17,1)',  # Fondo oscuro para el papel del gráfico
        font=dict(color='white'),  # Texto en blanco para mayor contraste
        title=dict(x=0.5, y=0.95),  # Ajustar la posición del título si es necesario
        xaxis=dict(color='white', showline=True, linewidth=2, linecolor='white'),
        yaxis=dict(color='white', showline=True, linewidth=2, linecolor='white'),
        legend=dict(
            font=dict(color='white'),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    # Actualizar las trazas para cambiar el color del texto a blanco y mejorar la visibilidad
    fig_school_type.update_traces(marker_line_color='white', marker_line_width=1.5)


    # Ahora, usa 'point_index' para obtener los valores SHAP para el punto seleccionado
    if point_index is not None:
        # Asumiendo que `explainer` ya está definido y es un TreeExplainer
        shap_values = explainer.shap_values(X_test.iloc[point_index])

        # Si el modelo es multiclase, `shap_values` será una lista de arrays
        if isinstance(shap_values, list):
            # Convertir shap_values en un array de numpy para facilitar el cálculo
            shap_values_array = np.array(shap_values)
            
            # Calcular el promedio de los valores SHAP a través de todas las clases para cada característica
            shap_values_mean = np.mean(shap_values_array, axis=0)

            # Aquí `shap_values_mean` es un array con el valor medio de SHAP para cada característica
        else:
            # Para modelos de clasificación binaria o regresión, shap_values ya es un array numpy
            shap_values_mean = shap_values

        # Crear DataFrame para visualización
        df_shap = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values_mean.flatten()  # Aplanar en caso de que shap_values_mean tenga más de 1 dimensión
        })

        # Ordenar y filtrar el DataFrame como antes
        df_shap_sorted = df_shap.sort_values(by='shap_value', ascending=False)
        
        # Filtra las variables relacionadas con 'Comunidad_Autonoma'
        # Asumiendo que los nombres empiezan con 'Comunidad_Autonoma' o 'Autonomous_Community'
        df_shap_filtered = df_shap[~df_shap['feature'].str.startswith(('Autonomous_Community'))]
        
        # Ordena las características por sus valores SHAP de mayor a menor importancia
        df_shap_sorted = df_shap_filtered.sort_values(by='shap_value', ascending=False)
        # Añadir las barras individualmente para poder controlar su color
        for index, row in df_shap_sorted.iterrows():
            # Define el color de cada barra individualmente dentro del bucle
            color = 'blue' if row['shap_value'] > 0 else 'red'
            
            fig_shap.add_trace(go.Bar(
                x=[row['shap_value']],
                y=[row['feature']],
                orientation='h',
                marker_color=color,  # Aplicar el color definido
                name=row['feature']  # Esto añadirá cada característica a la leyenda, puede omitirse si se prefiere no tener leyenda
            ))

        # Actualizar el layout del gráfico según necesites
        fig_shap.update_layout(
            title="SHAP Values for the Selected Point",
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
            yaxis={'categoryorder': 'total ascending'}  # Asegúrate de que el orden de las características sea el correcto
        )
    else:
        # Proporciona una figura vacía o con valores por defecto si no hay punto seleccionado
        fig_shap = px.bar(title="Select a point in the 3D graph to see SHAP values")


    print("Comunidad Autónoma seleccionada:", selected_community)
    print("Datos del clic:", clickData)

    if selected_community is None or selected_community == 'All':
        # Mostrar el mapa de España completo sin filtro específico
        fig_map = px.choropleth(
            geojson=spain_geojson,
            featureidkey="properties.name",
            locations=pd.Series([feature["properties"]["name"] for feature in spain_geojson['features']]),  # Utiliza los nombres de las comunidades del geoJSON
            color=np.ones(len(spain_geojson['features'])),  # Crea una serie de unos para tener un color uniforme
            color_continuous_scale="Viridis"
        )
    else:

        mapeo_comunidades = {
            "Andalusia": "Andalucia",
            "Aragon": "Aragon",
            "Asturias": "Asturias",
            "Balearic Islands": "Baleares",
            "Basque Country": "Pais Vasco",
            "Canary Islands": "Canarias",
            "Cantabria": "Cantabria",
            "Castile and Leon": "Castilla-Leon",
            "Castilla-La Mancha": "Castilla-La Mancha",
            "Catalonia": "Cataluña",
            "Ceuta": "Ceuta",
            "Extremadura": "Extremadura",
            "Galicia": "Galicia",
            "La Rioja": "La Rioja",
            "Madrid": "Madrid",
            "Melilla": "Melilla",
            "Murcia": "Murcia",
            "Navarre": "Navarra",
            "Valencian Community": "Valencia"
        }

        # Suponiendo que df_final ya tiene una columna 'Comunidad_Autonoma' con los nombres en inglés
        df_vis['Nombre_GeoJSON'] = df_vis['Community'].map(mapeo_comunidades)


        # Primero, asignamos un valor bajo a todas las filas
        df_vis['highlight'] = 1  # O cualquier valor constante bajo

        # Luego, asignamos un valor más alto a la comunidad autónoma seleccionada
        valor_resaltado = 10  # Este valor debe ser mayor que el valor bajo asignado anteriormente
        if selected_community != 'All':
            df_vis.loc[df_vis['Community'] == selected_community, 'highlight'] = valor_resaltado

        # Ahora, creamos el mapa coroplético usando la nueva columna 'highlight'
        fig_map = px.choropleth(
            df_vis,
            geojson=spain_geojson,
            locations='Nombre_GeoJSON',  # Asegúrate de que esta columna existe y coincide con los nombres en el GeoJSON
            featureidkey="properties.name",
            color='highlight',
            color_continuous_scale=["#e5e5e5", "#ff0000"]  
        )

    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig_3d, fig_repeat, fig_level, fig_gender, fig_school_type, fig_shap, fig_map

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=False)
