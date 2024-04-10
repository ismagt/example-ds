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
from sklearn.ensemble import RandomForestClassifier
import shap

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
            dbc.Col(dcc.Graph(id='shap-bar-chart'), width=4),
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
        dbc.Row([
            dbc.Col(dcc.Graph(id='histogram-had-class'), width=4),
            dbc.Col(dcc.Graph(id='histogram-books'), width=4),
            dbc.Col(dcc.Graph(id='histogram-num-devices'), width=4),
        ]),
    ],),
   html.Footer(
        [
            html.Div(
                [
                    html.P([
                        "Designed by Ismael Gómez-Talal",
                        html.Br(),
                        "(Member of BigMed+ Group)"
                    ], className="footer-text"),
                    html.Img(
                        src=app.get_asset_url('vertical.png'),
                        className="footer-image"
                    ),
                    html.A(
                    # Componente de imagen
                    html.Img(
                        src=app.get_asset_url('logo_urjc.png'),
                        className="footer-image",
                        style={'cursor': 'pointer'}
                    ),
                    # URL a la que redirigirá el clic en la imagen
                    href="https://gestion2.urjc.es/pdi/grupos-investigacion/bigmed+",
                    # Opcional: para abrir el enlace en una nueva pestaña
                    target="_blank"
                    ),
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
        Output('community-map', 'figure'),
        Output('histogram-had-class', 'figure'),
        Output('histogram-books', 'figure'),
        Output('histogram-num-devices', 'figure'),
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

    fig_had_extra_math_classes = go.Figure()
    fig_num_books = go.Figure()
    fig_num_digital_devices = go.Figure()
    fig_shap = go.Figure()

    # Define un mapa de colores para mantener los colores consistentes
    color_map = px.colors.qualitative.Plotly

    # Si seleccionas 'Todas', asegúrate de aplicar el mapa de colores
    if selected_community == 'All':
        for i, community in enumerate(df_vis['Community'].unique()):
            subset = df_vis[df_vis['Community'] == community]
            hover_text = subset.apply(lambda row: f"Autonomous_Community: {row['Community']}<br>" +
                                                    f"Repeat: {row['REPEAT']}<br>" +
                                                    f"Gender: {row['Gender']}<br>" +
                                                    f"School Type: {row['School_Type']}<br>" +
                                                    f"Level: {row['level']}<br>" +
                                                    f"Num digital devices: {row['Num_digital_devices']}<br>" +
                                                    f"Num books: {row['Num_books']}<br>" +
                                                    f"No extra math classes: {row['Had_extra_math_classes']}",
                                                    axis=1).tolist()
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
                                                    f"Repeat: {row['REPEAT']}<br>" +
                                                    f"Gender: {row['Gender']}<br>" +
                                                    f"School Type: {row['School_Type']}<br>" +
                                                    f"Level: {row['level']}<br>" +
                                                    f"Num digital devices: {row['Num_digital_devices']}<br>" +
                                                    f"Num books: {row['Num_books']}<br>" +
                                                    f"No extra math classes: {row['Had_extra_math_classes']}",
                                                    axis=1).tolist()

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

    # Definir la función de mapeo para los dispositivos digitales
    def categorize_devices(x):
        if x == "1":
            return "0"
        elif x == "2":
            return "1"
        elif x == "3":
            return "2"
        elif x == "4":
            return "3"
        elif x == "5":
            return "4"
        elif x == "6":
            return "5"
        elif x == "7":
            return "6-10"
        else:
            return "More than 10"

    # Aplicar la función para crear una nueva columna en el DataFrame
    filtered_df['Num_digital_devices_range'] = filtered_df['Num_digital_devices'].apply(categorize_devices)

    # Definir el orden de las categorías basado en la lógica deseada
    device_order = ['0', '1', '2', '3', '4', '5', '6-10', 'More than 10']

    # Calcular los conteos de cada categoría
    counts = filtered_df['Num_digital_devices_range'].value_counts(normalize=True) * 100  # los porcentajes
    counts = counts.reindex(device_order)  # asegurar que el orden es correcto

    # Crear el histograma con las categorías y el orden definidos
    fig_num_digital_devices = px.histogram(filtered_df, x='Num_digital_devices_range',
                                        category_orders={'Num_digital_devices_range': device_order},
                                        title="Number of Digital Devices with Screens in the Home",
                                        labels={'Num_digital_devices_range': "Number of Devices"},
                                        color_discrete_sequence=["#B6E2FA"])  # Color suave

    # Asegurarse de que el histograma muestre el orden deseado
    fig_num_digital_devices.update_xaxes(categoryorder='array', categoryarray=device_order)

    # Añadir anotaciones de porcentaje para cada barra
    for index, value in counts.items():
        fig_num_digital_devices.add_annotation(
            x=index,
            y=value,
            text=f"{value:.2f}%",  # Formato de porcentaje con dos decimales
            showarrow=False,
            yshift=10
        )

    # Actualizar el layout para evitar que las anotaciones queden fuera de los límites del gráfico
    fig_num_digital_devices.update_layout(margin={'t': 50, 'b': 50, 'r': 0, 'l': 0})


    # Suponiendo que 'filtered_df' tiene una columna 'Had_extra_math_classes' con valores 1 y 0
    counts = filtered_df['Had_extra_math_classes'].value_counts()

    # Cambiar las etiquetas a inglés. Asumiendo que 1 significa que el estudiante NO asiste a clases extra y 0 que SÍ asiste
    labels = ['Does not attend extra classes' if label == '1' else 'Attends extra classes' for label in counts.index]

    # Crear el gráfico de tarta con las etiquetas en inglés
    fig_had_extra_math_classes = go.Figure(data=[go.Pie(labels=labels, values=counts.values, hole=.3)])

    # Personalizar el gráfico de tarta
    fig_had_extra_math_classes.update_traces(textinfo='percent+label', marker=dict(colors=["#6495ED", "#FFD700"]))
    fig_had_extra_math_classes.update_layout(title_text="Extra Math Classes Attendance")

    # Ajustar el layout para dar espacio a las anotaciones de emoticonos, si decides añadirlas
    fig_had_extra_math_classes.update_layout(margin=dict(t=50, l=25, r=100, b=25))

    def categorize_books(x):
        if x == "1":
            return "0"
        elif x == "2":
            return "1-10"
        elif x == "3":
            return "11-25"
        elif x == "4":
            return "26-100"
        elif x == "5":
            return "101-200"
        elif x == "6":
            return "201-500"
        else:
            return "More than 500"

    # Aplicar la función para crear una nueva columna de rangos de libros
    filtered_df['Num_books_range'] = filtered_df['Num_books'].apply(categorize_books)

    # Ordenar las categorías manualmente
    order = ['0', '1-10', '11-25', '26-100', '101-200', '201-500', 'More than 500']

    # Calcular los conteos de cada categoría en porcentaje
    counts = filtered_df['Num_books_range'].value_counts(normalize=True) * 100  # los porcentajes
    counts = counts.reindex(order)  # asegurar que el orden es correcto

    # Crear el histograma con las categorías y el orden definidos
    fig_num_books = px.histogram(filtered_df, x='Num_books_range',
                                category_orders={'Num_books_range': order},
                                title="Number of Books in the House",
                                labels={'Num_books_range': "Number of Books"},
                                color_discrete_sequence=["#A9CCE3"])  # Color suave

    # Asegurarse de que el histograma muestre el orden deseado
    fig_num_books.update_xaxes(categoryorder='array', categoryarray=order)

    # Añadir anotaciones de porcentaje para cada barra
    for index, value in counts.items():  # Usar .items() en lugar de .iteritems()
        fig_num_books.add_annotation(
            x=index,
            y=value,
            text=f"{value:.2f}%",  # Formato de porcentaje con dos decimales
            showarrow=False,
            yshift=10
        )

    # Actualizar el layout para evitar que las anotaciones queden fuera de los límites del gráfico
    fig_num_books.update_layout(margin={'t': 50, 'b': 50, 'r': 0, 'l': 0})


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
                # Variables para almacenar valores positivos y negativos de SHAP por separado
        positive_shap = df_shap_sorted[df_shap_sorted['shap_value'] > 0]
        negative_shap = df_shap_sorted[df_shap_sorted['shap_value'] <= 0]

        # Añadir traza para valores SHAP positivos
        fig_shap.add_trace(go.Bar(
            x=positive_shap['shap_value'],
            y=positive_shap['feature'],
            orientation='h',
            marker_color='blue',
            name='Contributes to low level performance'
        ))

        # Añadir traza para valores SHAP negativos
        fig_shap.add_trace(go.Bar(
            x=negative_shap['shap_value'],
            y=negative_shap['feature'],
            orientation='h',
            marker_color='red',
            name='Contributes to high level performance'
        ))

        # Actualizar el layout del gráfico
        fig_shap.update_layout(
            title="SHAP Values for the Selected Point",
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
            yaxis={'categoryorder': 'total ascending'},
            legend_title="SHAP contribution"
        )
        # # Añadir las barras individualmente para poder controlar su color
        # for index, row in df_shap_sorted.iterrows():
        #     # Define el color de cada barra individualmente dentro del bucle
        #     color = 'blue' if row['shap_value'] > 0 else 'red'
            
        #     fig_shap.add_trace(go.Bar(
        #         x=[row['shap_value']],
        #         y=[row['feature']],
        #         orientation='h',
        #         marker_color=color,  # Aplicar el color definido
        #         name=row['feature']  # Esto añadirá cada característica a la leyenda, puede omitirse si se prefiere no tener leyenda
        #     ))

        # # Actualizar el layout del gráfico según necesites
        # fig_shap.update_layout(
        #     title="SHAP Values for the Selected Point",
        #     xaxis_title="SHAP Value",
        #     yaxis_title="Feature",
        #     yaxis={'categoryorder': 'total ascending'}  # Asegúrate de que el orden de las características sea el correcto
        # )
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

    return fig_3d, fig_repeat, fig_level, fig_gender, fig_school_type, fig_shap, fig_map, fig_had_extra_math_classes, fig_num_digital_devices, fig_num_books

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=False)
