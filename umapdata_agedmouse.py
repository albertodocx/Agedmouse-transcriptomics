"""
Script para realizar visualización UMAP con selección de valores de metadatos para AgedMouse
"""

import os
import sys
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PyQt5.QtWidgets import (QFileDialog, QDialog, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QListWidget, QComboBox, 
                             QDialogButtonBox, QListWidgetItem, QApplication)
from PyQt5.QtCore import Qt

def select_csv_file(parent=None):
    """
    Abrir diálogo para seleccionar archivo CSV usando PyQt5
    """
    file_path, _ = QFileDialog.getOpenFileName(
        parent, 
        "Selecciona el archivo CSV", 
        "", 
        "CSV files (*.csv);;All files (*)"
    )
    return file_path

def save_file_dialog(parent=None, file_type='png', default_name='umap'):
    """
    Abrir diálogo para guardar archivo con extensión específica
    """
    file_path, _ = QFileDialog.getSaveFileName(
        parent, 
        f"Guardar archivo {file_type.upper()}", 
        f"{default_name}.{file_type}", 
        f"{file_type.upper()} files (*.{file_type});(*.svg);All files (*)"
    )
    return file_path

def apply_metadata_filters(df, filters):
    """
    Aplicar filtros de metadatos al DataFrame
    """
    filtered_df = df.copy()
    
    for column, selected_values in filters.items():
        if column not in filtered_df.columns:
            print(f"Advertencia: La columna {column} no existe en el dataframe.")
            continue
        
        # Filtrar por los valores seleccionados
        filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
    
    return filtered_df

class UMAPMetadataFilterDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df
        self.metadata_filters = {}
        self.setWindowTitle('Filtros de Metadatos para UMAP')
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        
        # Layout principal
        layout = QVBoxLayout()
        
        # Crear pestañas para cada columna
        self.filter_tabs = QComboBox()
        layout.addWidget(QLabel('Seleccionar Columna para Filtrar:'))
        layout.addWidget(self.filter_tabs)
        
        # Lista de valores
        self.values_list = QListWidget()
        self.values_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(QLabel('Seleccionar Valores:'))
        layout.addWidget(self.values_list)
        
        # Resumen de filtros aplicados
        self.filter_summary = QListWidget()
        layout.addWidget(QLabel('Filtros Aplicados:'))
        layout.addWidget(self.filter_summary)
        
        # Botones de acción
        btn_layout = QHBoxLayout()
        add_filter_btn = QPushButton('Añadir Filtro')
        add_filter_btn.clicked.connect(self.add_filter)
        clear_filter_btn = QPushButton('Limpiar Filtros')
        clear_filter_btn.clicked.connect(self.clear_filters)
        btn_layout.addWidget(add_filter_btn)
        btn_layout.addWidget(clear_filter_btn)
        layout.addLayout(btn_layout)
        
        # Botones estándar
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Configurar columnas
        self.setup_columns()
        
        # Conectar cambio de columna
        self.filter_tabs.currentTextChanged.connect(self.update_values_list)
    
    def setup_columns(self):
        """Configurar las columnas disponibles para filtrar"""
        # Excluir columnas completamente numéricas
        filterable_columns = [
            col for col in self.df.columns 
            if not pd.api.types.is_numeric_dtype(self.df[col])
        ]
        
        self.filter_tabs.addItems(filterable_columns)
    
    def update_values_list(self, column):
        """Actualizar lista de valores para la columna seleccionada"""
        # Limpiar lista actual
        self.values_list.clear()
        
        # Obtener valores únicos de la columna
        unique_values = self.df[column].unique()
        
        # Añadir valores a la lista
        for value in sorted(unique_values, key=str):
            item = QListWidgetItem(str(value))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.values_list.addItem(item)
    
    def add_filter(self):
        """Añadir filtro actual a la lista de filtros"""
        column = self.filter_tabs.currentText()
        
        # Obtener valores seleccionados
        selected_values = []
        for index in range(self.values_list.count()):
            item = self.values_list.item(index)
            if item.checkState() == Qt.Checked:
                selected_values.append(item.text())
        
        if not selected_values:
            return
        
        # Añadir o actualizar filtro
        self.metadata_filters[column] = selected_values
        
        # Actualizar resumen de filtros
        self.filter_summary.clear()
        for col, values in self.metadata_filters.items():
            self.filter_summary.addItem(f"{col}: {', '.join(values)}")
    
    def clear_filters(self):
        """Limpiar todos los filtros"""
        self.metadata_filters.clear()
        self.filter_summary.clear()
        
        # Desmarcar todos los elementos en la lista de valores
        for index in range(self.values_list.count()):
            item = self.values_list.item(index)
            item.setCheckState(Qt.Unchecked)

def create_umap_visualization(csv_path, metadata_filters=None):
    """
    Crear visualización UMAP de datos de células con diálogos de guardado personalizados
    """
    try:
        # Cargar datos
        original_df = pd.read_csv(csv_path)
        final_cells_with_genes_cleaned = original_df.copy()
        
        # Aplicar filtros de metadatos si se proporcionan
        if metadata_filters:
            final_cells_with_genes_cleaned = apply_metadata_filters(final_cells_with_genes_cleaned, metadata_filters)
        
        # Usar solo columnas numéricas para UMAP
        numeric_columns = [col for col in final_cells_with_genes_cleaned.columns 
                           if pd.api.types.is_numeric_dtype(final_cells_with_genes_cleaned[col])]
        
        # Validar que hay columnas numéricas
        if not numeric_columns:
            print("Error: No hay columnas numéricas para realizar UMAP")
            return None
        
        # Extraer datos de expresión para TODAS las columnas numéricas
        expression_data = final_cells_with_genes_cleaned[numeric_columns]
        
        # Aplicar UMAP
        umap_model = umap.UMAP(n_components=2, random_state=42)
        umap_embeddings = umap_model.fit_transform(expression_data)
        
        # Crear DataFrame con coordenadas UMAP y metadatos
        # Identificar columnas de metadatos (excluyendo genes y columnas numéricas)
        metadata_columns = [
            col for col in final_cells_with_genes_cleaned.columns 
            if (not pd.api.types.is_numeric_dtype(final_cells_with_genes_cleaned[col])) and 
               (not col.startswith('gene_'))
        ]
        
        # Crear nuevo DataFrame solo con metadatos y coordenadas UMAP
        umap_df_clean = final_cells_with_genes_cleaned[metadata_columns].copy()
        umap_df_clean['UMAP_1'] = umap_embeddings[:, 0]
        umap_df_clean['UMAP_2'] = umap_embeddings[:, 1]
        
        # Abrir diálogo para guardar CSV
        csv_output_path = save_file_dialog(file_type='csv', default_name='umap_metadata')
        if not csv_output_path:
            print("Guardado de CSV cancelado.")
            return None
        
        # Guardar coordenadas UMAP y metadatos
        umap_df_clean.to_csv(csv_output_path, index=False)
        
        # Preparar visualización de clusters
        unique_clusters = final_cells_with_genes_cleaned['cluster_name'].unique()
        cluster_codes = final_cells_with_genes_cleaned['cluster_name'].astype('category').cat.codes
        num_clusters = len(unique_clusters)
        
        # Crear visualización mejorada con más espacio para leyenda
        plt.figure(figsize=(20, 12), dpi=300)  # Aumentar ancho para leyenda
        
        # Use predefined colormap and normalize explicitly
        cmap = plt.cm.get_cmap('tab20')
        norm = plt.Normalize(vmin=cluster_codes.min(), vmax=cluster_codes.max())
        
        # Crear un subplot para el scatter plot
        gs = plt.GridSpec(1, 2, width_ratios=[3, 1])  # Ancho del plot y la leyenda
        ax = plt.subplot(gs[0])
        
        scatter = ax.scatter(
            umap_embeddings[:, 0], 
            umap_embeddings[:, 1], 
            c=cluster_codes, 
            cmap=cmap, 
            norm=norm,
            s=20,  # Tamaño de punto ligeramente mayor
            alpha=0.7
        )
        
        # Personalizar plot
        ax.set_title('Visualización UMAP de Clusters Celulares\nFiltros de Metadatos Aplicados', fontsize=16)
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        
        # Crear subplot para leyenda
        legend_ax = plt.subplot(gs[1])
        legend_ax.axis('off')  # Desactivar ejes
        
        # Mejorar leyenda de clusters
        legend_elements = [
            plt.scatter([], [], color=cmap(norm(i)), 
                        label=cluster, s=100, alpha=0.7) 
            for i, cluster in enumerate(unique_clusters)
        ]
        legend_ax.legend(
            handles=legend_elements,
            title='Tipos Celulares', 
            title_fontsize=12,
            loc='center left',
            fontsize=10,
            markerscale=1.5
        )
        
        plt.tight_layout()
        
        # Abrir diálogo para guardar PNG
        png_output_path = save_file_dialog(file_type='png', default_name='umap_visualization')
        if not png_output_path:
            print("Guardado de PNG cancelado.")
            return None
        
        # Guardar figura
        plt.savefig(png_output_path, bbox_inches='tight')
        plt.close()
        
        # Imprimir información sobre el filtrado
        print(f"Total de células antes del filtrado: {len(original_df)}")
        print(f"Total de células después del filtrado: {len(final_cells_with_genes_cleaned)}")
        print(f"Coordenadas UMAP y metadatos guardados en: {csv_output_path}")
        print(f"Visualización UMAP guardada en: {png_output_path}")
        
        return png_output_path
    
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {csv_path}")
        return None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None

def main():
    # Crear una instancia de QApplication si no existe
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    # Para pruebas independientes
    csv_path = select_csv_file()
    if not csv_path:
        print("No se seleccionó ningún archivo.")
        return
    
    # Cargar DataFrame
    df = pd.read_csv(csv_path)
    
    # Mostrar diálogo de configuración de filtros
    config_dialog = UMAPMetadataFilterDialog(df)
    if config_dialog.exec_():
        # Obtener filtros
        filters = config_dialog.metadata_filters
        
        # Crear visualización UMAP
        output_path = create_umap_visualization(csv_path, filters)
        
        if output_path:
            print(f"Proceso completado con éxito")
        else:
            print("No se pudo crear la visualización UMAP")

if __name__ == "__main__":
    main()