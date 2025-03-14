import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO

# Función para exportar el gráfico como SVG
def export_plot(fig, format="svg"):
    img_bytes = pio.to_image(fig, format=format)
    return img_bytes

# Título de la aplicación
st.title("Visualización de UMAP de Expresión Génica")

# Sidebar para seleccionar archivo
st.sidebar.header("Carga de datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file, index_col=0)

    data = load_data(uploaded_file)

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

    # Sidebar de opciones de visualización
    st.sidebar.header("Opciones de visualización")
    filter_1 = st.sidebar.selectbox("Filtrar por primer elemento:", data.columns, index=list(data.columns).index('Cell type'))
    filter_2 = st.sidebar.selectbox("Filtrar por segundo elemento:", data.columns, index=list(data.columns).index('Sex'))
    filter_3 = st.sidebar.selectbox("Filtrar por tercer elemento:", data.columns, index=list(data.columns).index('Age'))
    color_filter = st.sidebar.selectbox("Seleccionar filtro para colorear:", ['None', filter_1, filter_2, filter_3])

    selected_filter_1 = st.sidebar.multiselect("Selecciona los valores para el primer filtro:", data[filter_1].unique())
    selected_filter_2 = st.sidebar.multiselect("Selecciona los valores para el segundo filtro:", data[filter_2].unique())
    selected_filter_3 = st.sidebar.multiselect("Selecciona los valores para el tercer filtro:", data[filter_3].unique())

    # Asignación de colores a los tipos de célula
    cell_types = data['Cell type'].unique()
    available_colors = ["#1F77B4", "#FF7F0E", "#2CA03C", "#D62728", "#9467BD", "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"]
    cell_type_color_mapping = {cell_type: available_colors[i % len(available_colors)] for i, cell_type in enumerate(cell_types)}

    # Aplicar los filtros
    filtered_data = data.copy()

    if color_filter != 'None':
        filtered_data['color'] = '#D3D3D3'

        selected_values = {filter_1: selected_filter_1, filter_2: selected_filter_2, filter_3: selected_filter_3}
        selected_indices = set(filtered_data.index)

        for col, values in selected_values.items():
            if values:
                selected_indices &= set(filtered_data[filtered_data[col].isin(values)].index)

        for group, group_data in filtered_data.loc[list(selected_indices)].groupby(color_filter):
            if color_filter == 'Sex':
                sex_color_mapping = {'Male': '#FFA500', 'Female': '#800080'}
                filtered_data.loc[group_data.index, 'color'] = sex_color_mapping.get(group, '#D3D3D3')
            elif color_filter == 'Age':
                age_color_mapping = {'Adult': '#FF00FF', 'Aged': '#00FFFF'}
                filtered_data.loc[group_data.index, 'color'] = age_color_mapping.get(group, '#D3D3D3')
            else:
                filtered_data.loc[group_data.index, 'color'] = cell_type_color_mapping.get(group, '#D3D3D3')
    else:
        filtered_data['color'] = '#90EE90'

    # Crear gráfico interactivo con Plotly
    fig = go.Figure()

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
        for group, group_data in filtered_data.groupby(color_filter):
            if not group_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=group_data['UMAP_1'],
                        y=group_data['UMAP_2'],
                        mode='markers',
                        marker=dict(color=group_data['color'], opacity=0.8),
                        name=str(group),
                        hoverinfo="text",
                        text=group_data[filter_1] + ', ' + group_data[filter_2] + ', ' + group_data[filter_3]
                    )
                )

    fig.update_layout(
        autosize=True,
        width=1200,
        height=800,
        title="UMAP de Expresión Génica"
    )

    # Mostrar el gráfico
    st.plotly_chart(fig)

    # Generar y descargar el gráfico en formato SVG
    img_bytes = export_plot(fig, format="svg")

    st.download_button(
        label="📥 Descargar gráfico como SVG",
        data=img_bytes,
        file_name="umap_gene_expression.svg",
        mime="image/svg+xml"
    )
else:
    st.warning("Por favor, sube un archivo Excel (.xlsx) para visualizar los datos.")

