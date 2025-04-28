"""Analisis de expresión genética por grupo con filtros Qt"""
import pandas as pd
import numpy as np
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDialog, QVBoxLayout, 
                           QLabel, QListWidget, QComboBox, QPushButton, QHBoxLayout, 
                           QDialogButtonBox, QMessageBox, QListWidgetItem)
from PyQt5.QtCore import Qt
from scipy import stats
from statsmodels.stats.multitest import multipletests

class MetadataColumnsDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df
        self.selected_metadata = []
        self.setWindowTitle('Selección de Columnas de Metadatos')
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        # Layout principal
        layout = QVBoxLayout()
        
        # Lista de columnas
        layout.addWidget(QLabel('Seleccione las columnas que son METADATOS (no genes):'))
        self.columns_list = QListWidget()
        self.columns_list.setSelectionMode(QListWidget.MultiSelection)
        
        # Añadir todas las columnas a la lista
        for column in df.columns:
            item = QListWidgetItem(column)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.columns_list.addItem(item)
        
        layout.addWidget(self.columns_list)
        
        # Botones de diálogo
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def accept_selection(self):
        # Obtener columnas seleccionadas
        for index in range(self.columns_list.count()):
            item = self.columns_list.item(index)
            if item.checkState() == Qt.Checked:
                self.selected_metadata.append(item.text())
        
        self.accept()

class MetadataFilterDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df
        self.metadata_filters = {}
        self.setWindowTitle('Filtros de Metadatos')
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        
        # Layout principal
        layout = QVBoxLayout()
        
        # Seleccionar columna para filtrar
        layout.addWidget(QLabel('Seleccionar Columna para Filtrar:'))
        self.column_combo = QComboBox()
        self.column_combo.addItems(list(df.columns))
        layout.addWidget(self.column_combo)
        
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
        
        # Conectar cambio de columna
        self.column_combo.currentTextChanged.connect(self.update_values_list)
        
        # Inicializar lista de valores
        self.update_values_list(self.column_combo.currentText())
    
    def update_values_list(self, column):
        """Actualizar lista de valores para la columna seleccionada"""
        # Limpiar lista actual
        self.values_list.clear()
        
        # Obtener valores únicos de la columna
        unique_values = self.df[column].dropna().unique()
        
        # Añadir valores a la lista
        for value in sorted(unique_values, key=str):
            item = QListWidgetItem(str(value))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.values_list.addItem(item)
    
    def add_filter(self):
        """Añadir filtro actual a la lista de filtros"""
        column = self.column_combo.currentText()
        
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

class GroupSelectionDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df
        self.setWindowTitle('Selección de Grupos')
        self.setMinimumWidth(400)
        self.selection = {}
        
        # Layout principal
        layout = QVBoxLayout()
        
        # Selector de columna
        layout.addWidget(QLabel('Seleccione una columna de filtro:'))
        self.column_combo = QComboBox()
        self.column_combo.addItems(list(df.columns))
        layout.addWidget(self.column_combo)
        
        # Selector Grupo 1
        layout.addWidget(QLabel('Seleccione un valor para el Grupo 1:'))
        self.group1_combo = QComboBox()
        layout.addWidget(self.group1_combo)
        
        # Selector Grupo 2
        layout.addWidget(QLabel('Seleccione un valor para el Grupo 2:'))
        self.group2_combo = QComboBox()
        layout.addWidget(self.group2_combo)
        
        # Botones de diálogo
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Conectar cambio de columna
        self.column_combo.currentTextChanged.connect(self.update_group_values)
        
        # Inicializar valores
        self.update_group_values(self.column_combo.currentText())
    
    def update_group_values(self, column):
        """Actualizar valores disponibles para grupos"""
        values = list(map(str, self.df[column].dropna().unique()))
        
        # Actualizar combo boxes
        self.group1_combo.clear()
        self.group2_combo.clear()
        
        self.group1_combo.addItems(values)
        self.group2_combo.addItems(values)
        
        # Seleccionar segundo valor por defecto si hay suficientes
        if len(values) > 1:
            self.group2_combo.setCurrentIndex(1)
    
    def accept_selection(self):
        self.selection = {
            "columna": self.column_combo.currentText(),
            "grupo1": self.group1_combo.currentText(),
            "grupo2": self.group2_combo.currentText()
        }
        self.accept()

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.setWindowTitle("Análisis de Expresión Genética")
        self.resize(800, 600)
        
        # Crear el menú y la barra de herramientas aquí si es necesario
        
        # Iniciar el análisis
        self.load_data()
    
    def load_data(self):
        # Cargar archivo CSV
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecciona un archivo CSV", "", "Archivos CSV (*.csv)")
        
        if not file_path:
            QMessageBox.warning(self, "Advertencia", "No se seleccionó ningún archivo. Ejecución cancelada.")
            sys.exit()
        
        print(f"📂 Archivo seleccionado: {file_path}")
        self.df = pd.read_csv(file_path, index_col=0)
        
        # Continuar con el flujo del análisis
        self.select_metadata_columns()
    
    def select_metadata_columns(self):
        # Seleccionar explícitamente qué columnas son metadatos
        dialog = MetadataColumnsDialog(self.df, self)
        result = dialog.exec_()
        
        if result != QDialog.Accepted:
            sys.exit()
        
        columnas_metadatos = dialog.selected_metadata
        print(f"🏷️ Columnas de metadatos seleccionadas: {columnas_metadatos}")
        
        # Filtro inicial de metadatos
        reply = QMessageBox.question(self, "Filtro Inicial", 
                                    "¿Desea usar todas las células sin filtrar?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.No:
            self.apply_metadata_filters(columnas_metadatos)
        else:
            self.select_groups(columnas_metadatos)
    
    def apply_metadata_filters(self, columnas_metadatos):
        dialog = MetadataFilterDialog(self.df, self)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            filtro_meta = dialog.metadata_filters
            
            # Aplicar filtros
            for columna_meta, valores_meta in filtro_meta.items():
                self.df = self.df[self.df[columna_meta].isin(valores_meta)]
            
            print(f"🔍 Datos filtrados por: {filtro_meta}")
            
            self.select_groups(columnas_metadatos)
        else:
            sys.exit()
    
    def select_groups(self, columnas_metadatos):
        dialog = GroupSelectionDialog(self.df, self)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            seleccion = dialog.selection
            self.perform_analysis(columnas_metadatos, seleccion)
        else:
            sys.exit()
            
    def perform_analysis(self, columnas_metadatos, seleccion):
        filtro_columna, valor_grupo1, valor_grupo2 = seleccion["columna"], seleccion["grupo1"], seleccion["grupo2"]
        
        celdas_grupo1 = self.df[self.df[filtro_columna] == valor_grupo1].index
        celdas_grupo2 = self.df[self.df[filtro_columna] == valor_grupo2].index
        
        print(f"📊 Filas en Grupo 1: {len(celdas_grupo1)}, Filas en Grupo 2: {len(celdas_grupo2)}")
        
        # Identificar las columnas de expresión de genes (excluyendo metadatos)
        columnas_de_genes = [col for col in self.df.columns if col not in columnas_metadatos]
        print(f"📊 Columnas de genes detectadas: {len(columnas_de_genes)}")
        
        # Inicializar listas para almacenar resultados
        p_values, t_values = [], []
        media_grupo1, media_grupo2, diff_expresion = [], [], []
        genes_analizados = []
        
        # Realizar la prueba t para cada gen
        for gen in columnas_de_genes:
            try:
                expression_grupo1 = pd.to_numeric(self.df.loc[celdas_grupo1, gen], errors='coerce')
                expression_grupo2 = pd.to_numeric(self.df.loc[celdas_grupo2, gen], errors='coerce')
                
                expression_grupo1 = expression_grupo1.dropna()
                expression_grupo2 = expression_grupo2.dropna()
                
                if len(expression_grupo1) > 1 and len(expression_grupo2) > 1:
                    mean_g1, mean_g2 = expression_grupo1.mean(), expression_grupo2.mean()
                    diff = mean_g1 - mean_g2
                    
                    t_stat, p_val = stats.ttest_ind(expression_grupo1, expression_grupo2, equal_var=False)
                    
                    # Solo añadir si el p-valor es un número válido (no NaN o infinito)
                    if not np.isnan(p_val) and not np.isinf(p_val):
                        genes_analizados.append(gen)
                        media_grupo1.append(mean_g1)
                        media_grupo2.append(mean_g2)
                        diff_expresion.append(diff)
                        t_values.append(t_stat)
                        p_values.append(p_val)
            except Exception as e:
                print(f"⚠️ Error al procesar el gen {gen}: {e}")
        
        print(f"🧬 Genes analizados correctamente: {len(genes_analizados)}")
        print(f"📊 Número de p-valores calculados: {len(p_values)}")
        
        # Ajustar los p-valores usando Benjamini-Hochberg
        if len(p_values) > 0:
            try:
                p_values_array = np.array(p_values)
                if np.any(np.isnan(p_values_array)) or np.any(np.isinf(p_values_array)):
                    print("⚠️ Advertencia: Hay valores NaN o infinitos en los p-valores")
                    valid_indices = ~(np.isnan(p_values_array) | np.isinf(p_values_array))
                    if np.sum(valid_indices) > 0:
                        valid_p_values = p_values_array[valid_indices]
                        reject, adj_p_values_partial, _, _ = multipletests(valid_p_values, method='fdr_bh', alpha=0.05)
                        
                        adj_p_values = np.full_like(p_values_array, np.nan)
                        adj_p_values[valid_indices] = adj_p_values_partial
                    else:
                        print("❌ No hay p-valores válidos para ajustar")
                        adj_p_values = np.full_like(p_values_array, np.nan)
                else:
                    reject, adj_p_values, _, _ = multipletests(p_values_array, method='fdr_bh', alpha=0.05)
            except Exception as e:
                print(f"❌ Error al calcular FDR: {e}")
                adj_p_values = np.full(len(p_values), np.nan)
        else:
            print("❌ No hay p-valores para ajustar")
            adj_p_values = []
        
        # Crear DataFrame de resultados
        resultados_df = pd.DataFrame({
            'Gene': genes_analizados,
            'Mean_Group1': media_grupo1,
            'Mean_Group2': media_grupo2,
            'Average_Expression': [(media_grupo1[i] + media_grupo2[i]) / 2 for i in range(len(genes_analizados))],
            'Diff_Expression': diff_expresion,
            'Log2_Fold_Change': [media_grupo1[i] - media_grupo2[i] for i in range(len(genes_analizados))],
            't': t_values,
            'P.Value': p_values,
            'adj.P.Val': adj_p_values.tolist() if isinstance(adj_p_values, np.ndarray) else adj_p_values,
            '"-log10_adj_P.Val"': [999 if p <= 0 else -np.log10(p) for p in (adj_p_values.tolist() if isinstance(adj_p_values, np.ndarray) else adj_p_values)]
        })
        
        # Ordenar por valor-p ajustado
        resultados_df = resultados_df.sort_values('adj.P.Val')
        
        # Guardar los resultados de expresión diferencial en un CSV
        output_file, _ = QFileDialog.getSaveFileName(self, "Guardar resultados de expresión diferencial", "", "Archivos CSV (*.csv)")
        if output_file:
            resultados_df.to_csv(output_file, index=False)
            print(f"✅ Resultados de expresión diferencial guardados en: {output_file}")
            QMessageBox.information(self, "Completado", f"Análisis completado. Resultados guardados en:\n{output_file}")
        else:
            print("⚠ No se guardaron los resultados de expresión diferencial.")
            QMessageBox.warning(self, "Advertencia", "No se guardaron los resultados de expresión diferencial.")
        
        print("✅ Análisis completado.")
        sys.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    sys.exit(app.exec_())