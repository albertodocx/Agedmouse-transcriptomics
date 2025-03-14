#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import IPython

os._exit(00)  # Cierra el proceso del kernel abruptamente
IPython.Application.instance().kernel.do_shutdown(True)  # Reinicia el kernel de manera controlada

import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import re
from pathlib import Path
import numpy as np
from tkinter import filedialog, Tk
from abc_atlas_access.abc_atlas_cache.anndata_utils import get_gene_data
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

# Configurar Tkinter
root = Tk()
root.withdraw()  # Ocultar ventana principal

# Configurar el directorio y caché
download_base = Path('../../data/abc_atlas')
abc_cache = AbcProjectCache.from_cache_dir(download_base)

# Obtener metadatos de genes
gene = abc_cache.get_metadata_dataframe(directory='WMB-10X', file_name='gene').set_index('gene_identifier')

# Obtener metadatos de células
cell_metadata = abc_cache.get_metadata_dataframe(
    directory='Zeng-Aging-Mouse-10Xv3',
    file_name='cell_metadata',
    dtype={'cell_label': str, 'wmb_cluster_alias': 'Int64'}
)

# Obtener anotaciones de clusters celulares
cell_cluster_annotations = abc_cache.get_metadata_dataframe(
    directory='Zeng-Aging-Mouse-10Xv3',
    file_name='cell_cluster_annotations'
)

# Fusionar datos
merged_cells = pd.merge(
    cell_metadata,
    cell_cluster_annotations[['cell_label', 'neurotransmitter_combined_label', 'cluster_name']],
    on='cell_label',
    how='inner'
)

# Normalizar nombres de cluster
final_cells_with_genes['cluster_name'] = final_cells_with_genes['cluster_name'].apply(
    lambda x: re.sub(r'^\d+_|_\d+$', '', x)
)

# Filtrar células
filtered_cells = merged_cells[
    (merged_cells['region_of_interest_label'] == 'HPF - HIP') &
    (merged_cells['neurotransmitter_combined_label'] == 'GABA')
]
filtered_cells = filtered_cells.set_index('cell_label')

# Obtener datos de expresión génica
gene_data = get_gene_data(
    abc_atlas_cache=abc_cache,
    all_cells=filtered_cells,
    all_genes=gene,
    selected_genes=gene['gene_symbol'].tolist(),
    data_type="log2"
)

# Validar datos
valid_cells = gene_data.index.intersection(filtered_cells.index)
valid_gene_data = gene_data.loc[valid_cells]
final_cells_with_genes = filtered_cells.merge(valid_gene_data, left_index=True, right_index=True)


# Selección de carpeta para guardar
output_directory = filedialog.askdirectory(title="Selecciona la carpeta para guardar archivos")
if not output_directory:
    raise ValueError("No se seleccionó ninguna carpeta para guardar")

# Guardar datos completos
output_file = filedialog.asksaveasfilename(
    title="Selecciona el nombre para guardar los datos completos",
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")]
)
if output_file:
    final_cells_with_genes.to_csv(output_file, index=True)

# Lista de columnas a eliminar
columns_to_remove = [
    "cell_barcode", "gene_count", "umi_count", "doublet_score", "x", "y", 
    "cluster_alias", "cell_in_wmb_study", "wmb_cluster_alias", "library_label", 
    "alignment_job_id", "library_method", "barcoded_cell_sample_label", 
    "enrichment_population", "library_in_wmb_study", "donor_label", 
    "population_sampling", "donor_genotype", "donor_age", "donor_in_wmb_study", 
    "feature_matrix_label", "dataset_label", "abc_sample_id", 
]

final_cells_with_genes_cleaned = final_cells_with_genes.drop(columns=columns_to_remove, errors='ignore')
output_directory = filedialog.askdirectory(title="Selecciona la carpeta para guardar archivos")
if not output_directory:
    raise ValueError("No se seleccionó ninguna carpeta para guardar")

output_file = filedialog.asksaveasfilename(
    title="Selecciona el nombre para guardar los datos completos limpios",
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")]
)
if output_file:
    final_cells_with_genes_cleaned.to_csv(output_file, index=True)


# Aplicar UMAP
expression_data = final_cells_with_genes_cleaned[gene['gene_symbol'].tolist()]
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(expression_data)

# Asignar colores a clusters
cluster_codes = final_cells_with_genes_cleaned['cluster_name'].astype('category').cat.codes
num_clusters = len(final_cells_with_genes_cleaned['cluster_name'].unique())
cmap = cm.get_cmap('tab20', num_clusters)

# Graficar UMAP
plt.figure(figsize=(12, 10))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=cluster_codes, cmap=cmap, s=5, alpha=0.7)
plt.title('UMAP de Expresión Génica con Tipos Celulares')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# Crear leyenda
cluster_labels = final_cells_with_genes_cleaned['cluster_name'].astype('category').cat.categories
legend_patches = [mpatches.Patch(color=cmap(i / num_clusters), label=label) for i, label in enumerate(cluster_labels)]
plt.legend(handles=legend_patches, title="Cluster Name", loc="upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

# Guardar datos reducidos
columns_to_remove = gene['gene_symbol'].tolist()
reduced_data = final_cells_with_genes_cleaned.drop(columns=columns_to_remove, errors='ignore')
reduced_data['UMAP_1'] = umap_embeddings[:, 0]
reduced_data['UMAP_2'] = umap_embeddings[:, 1]

output_reduced_file = filedialog.asksaveasfilename(
    title="Selecciona el nombre para guardar la matriz reducida",
    defaultextension=".xlsx",
    filetypes=[("Excel files", "*.xlsx")]
)
if output_reduced_file:
    reduced_data.to_excel(output_reduced_file, index=True)

print("Proceso completado con éxito.")


# In[ ]:




