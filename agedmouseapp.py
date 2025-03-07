import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random

# Cargar datos
@st.cache_data
def load_data():
    file_path = r'G:\Alberto\Transcriptomics\Results\reduced_umap_data.xlsx'
    return pd.read_excel(file_path, index_col=0)

data = load_data()

# Renombrar columnas para visualización
column_mapping = {
    'cluster_name': 'Cell type',
    'donor_sex': 'Sex',
    'donor_age_category': 'Age'
}
data = data.rename(columns=column_mapping)

# Reemplazar valores en las columnas Sex y Age
data['Sex'] = data['Sex'].replace({'M': 'Male', 'F': 'Female'})
data['Age'] = data['Age'].replace({'aged': 'Aged', 'adult': 'Adult'})

# Crear la interfaz
st.title("Visualización de UMAP de Expresión Génica")
st.sidebar.header("Opciones de visualización")

# Selección de filtros para los datos con valores por defecto
filter_1 = st.sidebar.selectbox("Filtrar por primer elemento:", data.columns, index=list(data.columns).index('Cell type'))
filter_2 = st.sidebar.selectbox("Filtrar por segundo elemento:", data.columns, index=list(data.columns).index('Sex'))
filter_3 = st.sidebar.selectbox("Filtrar por tercer elemento:", data.columns, index=list(data.columns).index('Age'))

# Selección de filtro para colorear, incluyendo opción 'None'
color_filter = st.sidebar.selectbox("Seleccionar filtro para colorear:", ['None', filter_1, filter_2, filter_3])

# Selección de valores para cada filtro (permitir múltiples selecciones)
selected_filter_1 = st.sidebar.multiselect("Selecciona los valores para el primer filtro:", data[filter_1].unique())
selected_filter_2 = st.sidebar.multiselect("Selecciona los valores para el segundo filtro:", data[filter_2].unique())
selected_filter_3 = st.sidebar.multiselect("Selecciona los valores para el tercer filtro:", data[filter_3].unique())

# Asignación de colores hexadecimales a los tipos de célula
cell_types = data['Cell type'].unique()

# Lista de colores hexadecimales sin los excluidos (por ejemplo, orange, purple, cyan, magenta)
excluded_colors = ['#FFA500', '#800080', '#00FFFF', '#FF00FF']  # Hexadecimal para orange, purple, cyan, magenta
all_colors = [
    "#D2691E", "#3CB371", "#8B0000", "#7B68EE", "#A52A2A", "#4682B4", "#9ACD32", "#8B4513", "#FF8C00", "#6A5ACD"
]

# Filtrar los colores no deseados
available_colors = [color for color in all_colors if color not in excluded_colors]

# Asegurarnos de que tenemos suficientes colores
if len(cell_types) > len(available_colors):
    raise ValueError("No hay suficientes colores disponibles para los tipos de célula.")

# Asignamos colores de manera fija
cell_type_color_mapping = {cell_type: available_colors[i] for i, cell_type in enumerate(cell_types)}

# Aplicar los filtros
filtered_data = data.copy()

# Si el filtro de color no es 'None', asignamos colores a los puntos
if color_filter != 'None':
    # Asignar colores manualmente a los puntos seleccionados
    filtered_data['color'] = '#D3D3D3'  # Asignamos gris claro a todos los puntos por defecto

    # Aplicar filtros
    selected_values = {
        filter_1: selected_filter_1,
        filter_2: selected_filter_2,
        filter_3: selected_filter_3
    }

    selected_indices = set(filtered_data.index)
    for col, values in selected_values.items():
        if values:
            selected_indices &= set(filtered_data[filtered_data[col].isin(values)].index)

    # Asignar colores a los puntos seleccionados
    for group, group_data in filtered_data.loc[list(selected_indices)].groupby(color_filter):
        if color_filter == 'Sex':
            sex_color_mapping = {'Male': '#FFA500', 'Female': '#800080'}  # Colores anteriores para Male/Female
            filtered_data.loc[group_data.index, 'color'] = sex_color_mapping.get(group, '#D3D3D3')
        elif color_filter == 'Age':
            age_color_mapping = {'Adult': '#FF00FF', 'Aged': '#00FFFF'}  # Colores anteriores para Adult/Aged
            filtered_data.loc[group_data.index, 'color'] = age_color_mapping.get(group, '#D3D3D3')
        else:
            filtered_data.loc[group_data.index, 'color'] = cell_type_color_mapping.get(group, '#D3D3D3')

else:
    # Si no se selecciona filtro de color, asignamos un color uniforme (gris claro) a todos los puntos
    filtered_data['color'] = '#90EE90'  # Color hex de lightgreen

# Crear gráfico interactivo con Plotly
fig = go.Figure()

# Si se seleccionó 'None' para el filtro de color, solo se dibuja un solo trace con el color uniforme
if color_filter == 'None':
    fig.add_trace(
        go.Scatter(
            x=filtered_data['UMAP_1'],
            y=filtered_data['UMAP_2'],
            mode='markers',
            marker=dict(color=filtered_data['color'], opacity=0.8),
            hoverinfo="text",
            text=filtered_data[filter_1] + ', ' + filtered_data[filter_2] + ', ' + filtered_data[filter_3],
            name='Sin filtro'
        )
    )
else:
    # Si se seleccionó un filtro de color, crear un trace por cada grupo único de ese filtro
    groups_to_include_in_legend = []  # Para almacenar los grupos visibles en la leyenda

    for group, group_data in filtered_data.groupby(color_filter):
        if not group_data.empty:  # Solo agregar los grupos que no están vacíos
            fig.add_trace(
                go.Scatter(
                    x=group_data['UMAP_1'],
                    y=group_data['UMAP_2'],
                    mode='markers',
                    marker=dict(color=group_data['color'], opacity=0.8),
                    name=str(group),  # La leyenda será el valor del grupo
                    hoverinfo="text",
                    text=group_data[filter_1] + ', ' + group_data[filter_2] + ', ' + group_data[filter_3]
                )
            )
            groups_to_include_in_legend.append(str(group))  # Agregar grupo a la lista de grupos visibles

    # Establecer la leyenda solo para los grupos visibles
    fig.update_layout(
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            tracegroupgap=0,
            title="Grupos",
            traceorder='normal',
            # Mostrar solo los grupos seleccionados
            groupclick="toggleitem",
        )
    )

# Ajustar tamaño del gráfico
fig.update_layout(
    autosize=True,
    width=1200,
    height=800,
    title="UMAP de Expresión Génica"
)

# Mostrar el gráfico
st.plotly_chart(fig)
