#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

def seleccionar_metadatos(df):
    def confirmar_seleccion():
        columna = columna_var.get()
        seleccionados[columna] = [valores_listbox.get(i) for i in valores_listbox.curselection()]
        root.quit()
    
    root = tk.Tk()
    root.title("Selección de Metadatos")
    
    seleccionados = {}
    tk.Label(root, text="Seleccione una columna de metadatos para filtrar:").pack()
    columna_var = tk.StringVar(root)
    columna_dropdown = ttk.Combobox(root, textvariable=columna_var, values=list(df.columns))
    columna_dropdown.pack()
    
    tk.Label(root, text="Seleccione valores (puede elegir múltiples con Ctrl o Shift):").pack()
    valores_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
    valores_listbox.pack()
    
    def actualizar_valores(*args):
        valores_listbox.delete(0, tk.END)
        valores = list(df[columna_var.get()].dropna().unique()) if columna_var.get() else []
        for v in valores:
            valores_listbox.insert(tk.END, v)
    
    columna_var.trace("w", actualizar_valores)
    
    tk.Button(root, text="Confirmar", command=confirmar_seleccion).pack()
    root.mainloop()
    root.destroy()
    
    return seleccionados

def seleccionar_opciones(df):
    def confirmar_seleccion():
        seleccionados.update({"columna": filtro_var.get(), "grupo1": grupo1_var.get(), "grupo2": grupo2_var.get()})
        root.quit()
    
    root = tk.Tk()
    root.title("Selección de Grupos")
    
    seleccionados = {}
    tk.Label(root, text="Seleccione una columna de filtro:").pack()
    filtro_var = tk.StringVar(root)
    filtro_dropdown = ttk.Combobox(root, textvariable=filtro_var, values=list(df.columns))
    filtro_dropdown.pack()
    
    tk.Label(root, text="Seleccione un valor para el Grupo 1:").pack()
    grupo1_var = tk.StringVar(root)
    grupo1_dropdown = ttk.Combobox(root, textvariable=grupo1_var)
    grupo1_dropdown.pack()
    
    tk.Label(root, text="Seleccione un valor para el Grupo 2:").pack()
    grupo2_var = tk.StringVar(root)
    grupo2_dropdown = ttk.Combobox(root, textvariable=grupo2_var)
    grupo2_dropdown.pack()
    
    def actualizar_valores(*args):
        valores = list(df[filtro_var.get()].dropna().unique()) if filtro_var.get() else []
        grupo1_dropdown["values"] = valores
        grupo2_dropdown["values"] = valores
    
    filtro_var.trace("w", actualizar_valores)
    
    tk.Button(root, text="Confirmar", command=confirmar_seleccion).pack()
    root.mainloop()
    root.destroy()
    
    return seleccionados

root = tk.Tk()
root.withdraw()
csv_file = filedialog.askopenfilename(title="Selecciona un archivo CSV", filetypes=[("Archivos CSV", "*.csv")])

if not csv_file:
    print("⚠ No se seleccionó ningún archivo. Ejecución cancelada.")
else:
    print(f"📂 Archivo seleccionado: {csv_file}")
    df = pd.read_csv(csv_file, index_col=0)
    
    usar_todo = messagebox.askyesno("Filtro Inicial", "¿Desea usar todas las células sin filtrar?")
    
    if not usar_todo:
        filtro_meta = seleccionar_metadatos(df)
        for columna_meta, valores_meta in filtro_meta.items():
            df = df[df[columna_meta].isin(valores_meta)]
        print(f"🔍 Datos filtrados por {columna_meta}: {valores_meta}")
    
    seleccion = seleccionar_opciones(df)
    filtro_columna, valor_grupo1, valor_grupo2 = seleccion["columna"], seleccion["grupo1"], seleccion["grupo2"]
    
    print(f"🔍 Valores únicos en {filtro_columna}: {df[filtro_columna].unique()}")
    
    celdas_grupo1 = df[df[filtro_columna] == valor_grupo1].index
    celdas_grupo2 = df[df[filtro_columna] == valor_grupo2].index
    
    print(f"📊 Filas en Grupo 1: {len(celdas_grupo1)}, Filas en Grupo 2: {len(celdas_grupo2)}")
    
    if celdas_grupo1.empty or celdas_grupo2.empty:
        print("⚠ No hay datos suficientes en uno o ambos grupos. Ejecución cancelada.")
    else:
        grupo1 = df.loc[celdas_grupo1]
        grupo2 = df.loc[celdas_grupo2]
        num_cols = df.shape[1]
        primer_col_expresion = next(i for i in range(num_cols) if df.dtypes.iloc[i] in ["float64", "int64"])
        
        media_grupo1 = grupo1.iloc[:, primer_col_expresion:].mean()
        media_grupo2 = grupo2.iloc[:, primer_col_expresion:].mean()
        genes_comunes = (media_grupo1 > 0) & (media_grupo2 > 0)
        df_comunes = pd.DataFrame({valor_grupo1: media_grupo1[genes_comunes], valor_grupo2: media_grupo2[genes_comunes]})
        
        output_folder = filedialog.askdirectory(title="Selecciona la carpeta para guardar resultados")
        if output_folder:
            output_file = f"{output_folder}/resultado_{filtro_columna}_genes_comunes_{valor_grupo1}_vs_{valor_grupo2}.csv"
            df_comunes.to_csv(output_file)
            print(f"✅ Archivo guardado en: {output_file}")
        else:
            print("⚠ No se guardó ningún archivo.")
        
        print("✅ Análisis completado.")


# In[ ]:




