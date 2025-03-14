#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import time

# Crear ventana oculta de tkinter
root = tk.Tk()
root.withdraw()  # Ocultar ventana

# Selección interactiva del archivo CSV
file_path = filedialog.askopenfilename(title="Selecciona un archivo CSV", filetypes=[("Archivos CSV", "*.csv")])

# Verificar si se seleccionó un archivo
if not file_path:
    print("No se seleccionó ningún archivo. Ejecución cancelada.")
else:
    print(f"📂 Archivo seleccionado: {file_path}")
    
    # Cargar el archivo CSV
    df = pd.read_csv(file_path, index_col=0)
    print(df.columns)  # Muestra los nombres de las columnas
    print(df.shape)    # Muestra cuántas filas y columnas hay

    # Calcular la diferencia entre la expresión en Grupo1 y Grupo2
    df['Expression_Diff'] = df.iloc[:, 0] - df.iloc[:, 1]
    
    # Calcular media y desviación estándar
    mean_diff = df['Expression_Diff'].mean()
    std_diff = df['Expression_Diff'].std()
    
    # Definir umbrales de outliers extremos usando desviaciones estándar
    extreme_high = mean_diff + 5 * std_diff
    extreme_low = mean_diff - 5 * std_diff
    
    # Identificar los outliers extremos
    high_outliers = df[df['Expression_Diff'] > extreme_high].copy()
    low_outliers = df[df['Expression_Diff'] < extreme_low].copy()
    
    # Identificar outliers según IQR
    Q1 = df['Expression_Diff'].quantile(0.25)
    Q3 = df['Expression_Diff'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = df[(df['Expression_Diff'] < lower_bound) | (df['Expression_Diff'] > upper_bound)].copy()
    
    # Clasificar los outliers como "Positivo" o "Negativo"
    high_outliers['Outlier_Type'] = 'Positivo'
    low_outliers['Outlier_Type'] = 'Negativo'
    
    # Mostrar los genes que son outliers con su clasificación
    print("Genes considerados outliers extremos:")
    print(high_outliers[['Expression_Diff', 'Outlier_Type']])
    print(low_outliers[['Expression_Diff', 'Outlier_Type']])
    
    # Crear una columna de categoría para los outliers
    df['Category'] = 'Normal'
    df.loc[iqr_outliers.index, 'Category'] = 'Outlier IQR'
    df.loc[high_outliers.index, 'Category'] = 'Extremo Alto'
    df.loc[low_outliers.index, 'Category'] = 'Extremo Bajo'
    
    
    # Seleccionar dónde guardar el archivo de salida
    save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("Archivos CSV", "*.csv")],
                                             title="Guardar archivo como")
    
    if save_path:  # Verificar si el usuario eligió un archivo
        df.to_csv(save_path)
        print(f"✅ Archivo guardado en: {save_path}")
        
        # Guardar gráficos en el mismo directorio
        save_dir = "".join(save_path.rsplit("/", 1)[:-1])
        
        # Crear el gráfico de caja y bigotes
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=df['Expression_Diff'])
        plt.axhline(y=extreme_high, color='red', linestyle='dashed', label=f'5σ Alto ({extreme_high:.2f})')
        plt.axhline(y=extreme_low, color='blue', linestyle='dashed', label=f'5σ Bajo ({extreme_low:.2f})')
        plt.legend()
        plt.title('Distribución de la Diferencia de Expresión')
        plt.text(0.95, 0.95, f'SD = {std_diff:.2f}', horizontalalignment='center', verticalalignment='center', 
                 transform=plt.gca().transAxes, fontsize=12, color='green')
        plt.savefig(f"{save_dir}/boxplot.png")
        plt.close()
        
        # Crear el heatmap con la muestra representativa
        df_iqr = df[(df['Expression_Diff'] > lower_bound) & (df['Expression_Diff'] < upper_bound) & (df['Expression_Diff'] > extreme_low) & (df['Expression_Diff'] < extreme_high)]
        df_sample = df_iqr.sample(n=min(50, len(df_iqr)), random_state=42)
        df_sorted = df_sample.sort_values(by='Expression_Diff', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.heatmap(df_sorted[['Expression_Diff']].T, cmap='coolwarm', annot=False, xticklabels=df_sample.index)
        plt.xticks(rotation=90)
        plt.title('Diferencia de Expresión de Genes')
        plt.savefig(f"{save_dir}/heatmap.png")
        plt.close()
        
        print("✅ Gráficos guardados correctamente.")
    else:
        print("⚠ No se guardó ningún archivo.")
    
    print("✅ Análisis completado correctamente.")


# In[ ]:




