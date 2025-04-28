
import numpy as np

def custom_transform(x):
    """
    Custom transformation function to enhance visualization of differential expression.
    
    Parameters:
    x (float): The original differential expression value
    
    Returns:
    float: Transformed value that preserves sign and highlights magnitude differences
    """
    # Preserve the sign of the original value
    sign = np.sign(x)
    
    # Take the absolute value of the input
    abs_val = np.abs(x)
    
    # Apply square root transformation to enhance differences
    # This will compress larger values more than smaller values
    return sign * np.sqrt(abs_val)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import base64
import os

def main():
    st.set_page_config(layout="wide", page_title="AgedMouse Data Visualization")
    st.title("AgedMouse Data Visualization")

    # Sidebar for file upload
    st.sidebar.header("Upload Data Files")
    
    # UMAP Data File Upload
    umap_file = st.sidebar.file_uploader("Choose UMAP CSV File", type="csv")
    
    # Expression Data File Upload
    expression_file = st.sidebar.file_uploader("Choose Expression CSV File", type="csv")
    
    # Visualization section
    if umap_file and expression_file:
        # Read the uploaded files
        try:
            umap_df = pd.read_csv(umap_file)
            expression_df = pd.read_csv(expression_file)
            
            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["UMAP Visualization", "Gene Expression Graphs"])

            with tab1:
                st.header("UMAP Scatter Plot")
                data = umap_df.copy()  # This correctly assigns umap_df to data
                # Diseño
                # Rename columns for visualization
                column_mapping = {
                    'cluster_name': 'Cell type',
                    'donor_sex': 'Sex',
                    'donor_age_category': 'Age'
                }
                data = data.rename(columns=column_mapping)

                # Replace values in Sex and Age columns
                data['Sex'] = data['Sex'].replace({'M': 'Male', 'F': 'Female'})
                data['Age'] = data['Age'].replace({'aged': 'Aged', 'adult': 'Adult'})

                # Sidebar visualization options
                st.sidebar.header("Opciones de visualización")
                
                # Dynamically get columns (excluding UMAP coordinates)
                visualization_columns = [col for col in data.columns if col not in ['UMAP_1', 'UMAP_2']]
                
                filter_1 = st.sidebar.selectbox("Filtrar por primer elemento:", visualization_columns, index=0)
                filter_2 = st.sidebar.selectbox("Filtrar por segundo elemento:", visualization_columns, index=1 if len(visualization_columns) > 1 else 0)
                filter_3 = st.sidebar.selectbox("Filtrar por tercer elemento:", visualization_columns, index=2 if len(visualization_columns) > 2 else 0)
                
                color_filter = st.sidebar.selectbox("Seleccionar filtro para colorear:", ['None'] + visualization_columns)

                # Multiselect filters
                selected_filter_1 = st.sidebar.multiselect(f"Selecciona los valores para {filter_1}:", data[filter_1].unique())
                selected_filter_2 = st.sidebar.multiselect(f"Selecciona los valores para {filter_2}:", data[filter_2].unique())
                selected_filter_3 = st.sidebar.multiselect(f"Selecciona los valores para {filter_3}:", data[filter_3].unique())

                # Color mapping
                available_colors = ["#1F77B4", "#FF7F0E", "#2CA03C", "#D62728", "#9467BD", "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"]
                
                # Prepare filtered data
                filtered_data = data.copy()
                
                # Apply filters
                if selected_filter_1:
                    filtered_data = filtered_data[filtered_data[filter_1].isin(selected_filter_1)]
                if selected_filter_2:
                    filtered_data = filtered_data[filtered_data[filter_2].isin(selected_filter_2)]
                if selected_filter_3:
                    filtered_data = filtered_data[filtered_data[filter_3].isin(selected_filter_3)]

                # Color mapping logic
                filtered_data['color'] = '#D3D3D3'  # Default grey for "unchosen" data
                filtered_data['group'] = 'Unchosen'

                if color_filter == 'None':
                    filtered_data['color'] = '#000000'  # Negro para todos los puntos seleccionados
                    filtered_data['group'] = 'All Data'
                else:
                    unique_groups = filtered_data[color_filter].unique()
                    color_mapping = {group: available_colors[i % len(available_colors)] for i, group in enumerate(unique_groups)}
                    
                    # Special handling for Sex and Age
                    if color_filter == 'Sex':
                        color_mapping = {'Male': '#FFA500', 'Female': '#800080'}
                    elif color_filter == 'Age':
                        color_mapping = {'Adult': '#FF00FF', 'Aged': '#00FFFF'}

                    # Aplicar colores a los datos filtrados
                    for group, color in color_mapping.items():
                        mask = filtered_data[color_filter] == group
                        filtered_data.loc[mask, 'color'] = color
                        filtered_data.loc[mask, 'group'] = group

                # Crear figura de Plotly
                fig = go.Figure()

                # Agregar datos no seleccionados (puntos grises)
                unchosen_data = data[~data.index.isin(filtered_data.index)]
                fig.add_trace(
                    go.Scatter(
                        x=unchosen_data['UMAP_1'],
                        y=unchosen_data['UMAP_2'],
                        mode='markers',
                        marker=dict(color='#D3D3D3', opacity=0.3),
                        name='Unchosen',
                        hoverinfo="text",
                        text=unchosen_data[filter_1] + ', ' + unchosen_data[filter_2] + ', ' + unchosen_data[filter_3]
                    )
                )

                # Agregar datos filtrados y coloreados
                if color_filter == 'None':
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_data['UMAP_1'],
                            y=filtered_data['UMAP_2'],
                            mode='markers',
                            marker=dict(color='#000000', opacity=0.8),  # Negro para todos los seleccionados
                            name="All Data",
                            hoverinfo="text",
                            text=filtered_data[filter_1] + ', ' + filtered_data[filter_2] + ', ' + filtered_data[filter_3]
                        )
                    )
                else:
                    for group in filtered_data[color_filter].unique():
                        group_data = filtered_data[filtered_data[color_filter] == group]
                        fig.add_trace(
                            go.Scatter(
                                x=group_data['UMAP_1'],
                                y=group_data['UMAP_2'],
                                mode='markers',
                                marker=dict(color=color_mapping[group], opacity=0.8),
                                name=str(group),
                                hoverinfo="text",
                                text=group_data[filter_1] + ', ' + group_data[filter_2] + ', ' + group_data[filter_3]
                            )
                        )

                fig.update_layout(
                    title="UMAP de Expresión Génica",
                    xaxis_title="UMAP1",
                    yaxis_title="UMAP2",
                    height=1200,
                    width=1200
                )

                st.plotly_chart(fig)
                
                # UMAP Plot Download
                umap_format = st.selectbox("Select UMAP Plot Format", ["png", "svg", "jpg"], key="umap_format")
                if st.button(f"Download UMAP Plot ({umap_format})"):
                    umap_bytes = fig.to_image(format=umap_format)
                    b64 = base64.b64encode(umap_bytes).decode()
                    href = f'<a href="data:image/{umap_format};base64,{b64}" download="umap_plot.{umap_format}">Download UMAP Plot</a>'
                    st.markdown(href, unsafe_allow_html=True)

            with tab2:
                st.header("Gene Expression Analysis")

                # Calcular la diferencia entre medias de los dos grupos (mean group 1 y mean group 2)
                expression_df['Mean Difference'] = expression_df['Mean_Group1'] - expression_df['Mean_Group2']

                # Ordenar los genes por la magnitud de la diferencia (de mayor a menor)
                expression_df = expression_df.sort_values(by='Mean Difference', ascending=False)

                # Seleccionar los genes con sobreexpresión/subexpresión basados en Log2 Fold Change y p-valor
                log2_fc_threshold = st.slider(
                    "Select Log2 Fold Change Threshold", 
                    min_value=0.0, 
                    max_value=5.0, 
                    value=1.0, 
                    step=0.1,
                    help="Select the Log2 Fold Change threshold for overexpressed/subexpressed genes."
                )

                p_value_threshold = st.slider(
                    "Select P-Value Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.05, 
                    step=0.01,
                    help="Select the P-value threshold for statistically significant genes."
                )

                # Filtrar los genes que están sobre o subexpresados basados en Log2_Fold_Change y P-valor
                filtered_genes = expression_df#[
                   # (expression_df['Log2_Fold_Change'].abs() > log2_fc_threshold) & 
                   # (expression_df['P.Value'] < p_value_threshold)
               # ]

                # Verifica si hay genes filtrados
                if not filtered_genes.empty:
                    # Crear figura con subplots horizontales
                    fig = make_subplots(
                        rows=1, cols=2, 
                        subplot_titles=("Mean Expression", "Differential Expression"),
                        
                        horizontal_spacing=0.3
                    )

                    # Heatmap para Mean Groups
                    fig.add_trace(
                        go.Heatmap(
                            z=filtered_genes[['Mean_Group1', 'Mean_Group2']].values,
                            x=['Mean Group 1', 'Mean Group 2'],
                            y=filtered_genes['Gene'],
                            colorscale='RdBu_r',
                            colorbar=dict(title='Mean Expression', x=0.4)
                        ),
                        row=1, col=1
                    )
                    transformed_diff = filtered_genes['Diff_Expression'].apply(custom_transform)
                    # Heatmap para Differential Expression
                    fig.add_trace(
                        go.Heatmap(
                            z=transformed_diff.values.reshape(-1, 1),
                            x=['Diff Expression'],
                            y=filtered_genes['Gene'],
                            colorscale='RdBu_r',
                            colorbar=dict(title='Diff Expression', x=1)
                        ),
                        row=1, col=2
                    )

                    # Configurar layout
                    fig.update_layout(
                        title='Gene Expression Analysis',
                        height=1200,  # Ajustar altura dinámicamente
                        width=1200,
                        title_x=0.5
                    )

                    # Ajustar ejes
                    fig.update_xaxes(title_text='Groups', row=1, col=1)
                    fig.update_xaxes(title_text='Diff Expression', row=1, col=2)
                    fig.update_yaxes(title_text='Genes', row=1, col=1)

                    # Mostrar gráfico
                    st.plotly_chart(fig, use_container_width=True)

                
                    # Selector de formato de descarga
                    heatmap_format = st.selectbox(
                        "Select Heatmap Format", 
                        ["png", "svg", "jpg"], 
                        key="heatmap_format_combined"
                    )

                    # Botón de descarga
                    if st.button(f"Download Heatmap ({heatmap_format})"):
                        heatmap_bytes = fig.to_image(format=heatmap_format)
                        b64 = base64.b64encode(heatmap_bytes).decode()
                        href = f'<a href="data:image/{heatmap_format};base64,{b64}" download="gene_expression_heatmap.{heatmap_format}">Download Heatmap</a>'
                        st.markdown(href, unsafe_allow_html=True)

                else:
                    st.write(f"No genes found meeting Log2_Fold_Change ({log2_fc_threshold}) and p-value ({p_value_threshold}) thresholds.")
                    
        except Exception as e:
            st.error(f"Error processing files: {e}")
    else:
        st.warning("Please upload both UMAP and Expression data CSV files")

if __name__ == "__main__":
    main()


   