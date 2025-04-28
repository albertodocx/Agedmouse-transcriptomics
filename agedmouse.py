import sys
import os
import traceback
import io
import contextlib
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QListWidget, QPushButton, QWidget, QLabel, QTextEdit)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

class MultiScriptApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Obtener el directorio del script actual
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('AgedMouse Transcriptomics')
        self.setGeometry(100, 100, 1000, 700)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Logo
        logo_label = QLabel()
        logo_path = os.path.join(self.script_dir, 'agedmouse_logo.png')
        
        # Intentar cargar la imagen
        try:
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaledToWidth(800, Qt.SmoothTransformation))
            logo_label.setAlignment(Qt.AlignCenter)
        except Exception as e:
            print(f"Error cargando logo: {e}")
        
        main_layout.addWidget(logo_label)
        
        # Layout para scripts
        scripts_layout = QHBoxLayout()
        
        # Definir scripts disponibles
        self.scripts = {
            'Open Data': 'opendata_agedmouse.py',
            'UMAP Data': 'umapdata_agedmouse.py',
            'Expression Analysis': 'expressionanalysis_agedmouse.py',
            'UMAP Analysis':'umapstatisticalanalysis_agedmouse.py',
            'Visualization': 'visualization_agedmouse.py'
        }
        
        # Crear botones personalizados
        for script_name, script_file in self.scripts.items():
            btn = QPushButton(script_name)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    font-size: 16px;
                    margin: 4px 2px;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            btn.clicked.connect(lambda checked, sf=script_file: self.run_script(sf))
            scripts_layout.addWidget(btn)
        
        # Agregar layout de scripts al layout principal
        main_layout.addLayout(scripts_layout)
        
        # Layout para ejecución
        execution_layout = QVBoxLayout()
        
        # Descripción del script
        self.script_description = QLabel('Selecciona un script')
        execution_layout.addWidget(self.script_description)
        
        # Área de salida
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        execution_layout.addWidget(self.output_text)
        
        # Agregar layout de ejecución al layout principal
        main_layout.addLayout(execution_layout)
        
    def run_script(self, script_filename):
        try:
            # Específicamente para el script de visualización
            if script_filename == 'visualization_agedmouse.py':
                # Preparar el comando de Streamlit
                streamlit_cmd = [
                    sys.executable, 
                    '-m', 
                    'streamlit', 
                    'run', 
                    os.path.join(self.script_dir, script_filename)
                ]
                
                # Ejecutar Streamlit en un proceso separado
                subprocess.Popen(streamlit_cmd)
                
                # Actualizar el texto de salida
                self.output_text.setText(f"Launching Streamlit visualization for {script_filename}")
                return

            # Lógica existente para ejecutar otros scripts
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                # Ejecutar el script directamente con manejo de codificación
                script_path = os.path.join(self.script_dir, script_filename)
                with open(script_path, 'r', encoding='utf-8', errors='ignore') as file:
                    script_content = file.read()
                    exec(script_content, {'__name__': '__main__'})
            
            # Mostrar la salida
            self.output_text.setText(output.getvalue())
        
        except Exception as e:
            # Mostrar el traceback completo para depuración
            error_output = traceback.format_exc()
            self.output_text.setText(f'Error al ejecutar el script:\n{error_output}')

def main():
    app = QApplication(sys.argv)
    main_window = MultiScriptApp()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()