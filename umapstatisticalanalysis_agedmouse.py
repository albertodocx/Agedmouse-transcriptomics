import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import stats
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
import itertools
from pathlib import Path
from sklearn import metrics
from scipy.spatial import ConvexHull
import colorsys
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests

def select_file():
    """
    Abre un diálogo para seleccionar un archivo CSV
    """
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de tkinter
    
    file_path = filedialog.askopenfilename(
        title="Selecciona el archivo de datos",
        filetypes=[("Archivos CSV/TSV", "*.csv *.tsv *.txt"), 
                  ("Todos los archivos", "*.*")]
    )
    
    root.destroy()
    return file_path

def load_data(file_path):
    """
    Carga los datos desde el archivo seleccionado
    Detecta automáticamente el delimitador
    """
    try:
        # Primero intenta con tabulador (TSV)
        data = pd.read_csv(file_path, sep='\t')
        if len(data.columns) <= 1:  # Si solo detecta una columna, prueba con coma
            data = pd.read_csv(file_path, sep=',')
        
        return data
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def generate_highly_distinct_colors(n):
    """
    Genera colores altamente contrastados usando el espacio de color HSV
    para maximizar la distinción visual entre grupos
    """
    # Colores predefinidos de alto contraste para los primeros grupos
    distinct_colors = [
        '#FF0000',  # Rojo
        '#0000FF',  # Azul
        '#00CC00',  # Verde
        '#FF00FF',  # Magenta
        '#FFD700',  # Oro
        '#00FFFF',  # Cian
        '#FF6600',  # Naranja
        '#8B00FF',  # Violeta
        '#006400',  # Verde oscuro
        '#8B0000',  # Rojo oscuro
        '#4B0082',  # Índigo
        '#556B2F',  # Verde oliva
        '#800080',  # Púrpura
        '#008B8B',  # Cian oscuro
        '#B22222',  # Ladrillo
        '#4682B4',  # Azul acero
    ]
    
    if n <= len(distinct_colors):
        return distinct_colors[:n]
    else:
        # Si necesitamos más colores, generamos utilizando HSV con máxima separación
        colors = []
        for i in range(n):
            # Calcular el matiz con separación máxima
            hue = i / n
            # Usar alta saturación y valor para mejorar contraste
            saturation = 0.9
            value = 0.95
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convertir a formato hexadecimal
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors

def assign_distinct_category_colors(data, feature):
    """
    Asigna colores distintivos a las categorías, asegurando que categorías
    con nombres similares (como M/F y Viejo/Joven) tengan colores muy diferentes
    """
    unique_values = data[feature].unique()
    
    # Obtener colores base altamente distintivos
    colors_list = generate_highly_distinct_colors(len(unique_values))
    
    # Mezclar la lista de colores para evitar que categorías adyacentes 
    # tengan colores similares
    np.random.shuffle(colors_list)
    
    # Asignar colores
    colors = {val: colors_list[i] for i, val in enumerate(unique_values)}
    
    return colors

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Crea una elipse de confianza para los datos dados
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    
    # Verificar si la matriz de covarianza es válida
    if np.isnan(cov).any() or np.isinf(cov).any() or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        print("Advertencia: Matriz de covarianza no válida, no se puede dibujar la elipse")
        return None
    
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Eigenvalues y eigenvectors
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, 
                      facecolor=facecolor, **kwargs)
    
    # Escalar según desviación estándar
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    # Calcular medias para la posición
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    ellipse.set_alpha(0.3)  # Aumentado ligeramente para mejor visibilidad
    
    return ax.add_patch(ellipse)

def calculate_bivariate_measures(x, y):
    """
    Calcula métricas bivariadas para un conjunto de coordenadas 2D
    """
    if len(x) < 3:  # Necesitamos al menos 3 puntos para análisis significativo
        return {
            'mahalanobis_center': np.nan,
            'convex_hull_area': np.nan,
            'avg_distance_to_centroid': np.nan,
            'spatial_median_distance': np.nan,
            'spatial_dispersion': np.nan
        }
    
    points = np.column_stack((x, y))
    centroid = np.mean(points, axis=0)
    
    # Matriz de covarianza
    cov = np.cov(x, y)
    if np.linalg.det(cov) <= 0:  # Verificar si es invertible
        mahalanobis_center = np.nan
    else:
        # Distancia de Mahalanobis desde cada punto al centro
        try:
            cov_inv = np.linalg.inv(cov)
            mahalanobis_distances = []
            for pt in points:
                diff = pt - centroid
                mahalanobis_dist = np.sqrt(diff @ cov_inv @ diff.T)
                mahalanobis_distances.append(mahalanobis_dist)
            mahalanobis_center = np.mean(mahalanobis_distances)
        except:
            mahalanobis_center = np.nan
    
    # Área del casco convexo (espacio ocupado por los puntos)
    try:
        hull = ConvexHull(points)
        convex_hull_area = hull.volume  # En 2D, volumen = área
    except:
        convex_hull_area = np.nan
    
    # Distancia euclidiana promedio al centroide
    distances_to_centroid = [np.linalg.norm(pt - centroid) for pt in points]
    avg_distance_to_centroid = np.mean(distances_to_centroid)
    
    # Mediana espacial (punto que minimiza suma de distancias)
    # Aproximamos usando la mediana de cada coordenada
    spatial_median = np.median(points, axis=0)
    spatial_median_distance = np.mean([np.linalg.norm(pt - spatial_median) for pt in points])
    
    # Dispersión espacial (varianza promedio de los puntos)
    spatial_dispersion = np.mean([np.linalg.norm(pt - centroid)**2 for pt in points])
    
    return {
        'mahalanobis_center': mahalanobis_center,
        'convex_hull_area': convex_hull_area,
        'avg_distance_to_centroid': avg_distance_to_centroid,
        'spatial_median_distance': spatial_median_distance,
        'spatial_dispersion': spatial_dispersion
    }

def analyze_clusters(data, feature, x_col, y_col, ax=None, show_ellipses=True):
    """
    Analiza los clusters basados en el feature especificado
    Grafica en el eje proporcionado o crea una nueva figura
    Retorna centroides y distancias
    """
    create_new_fig = (ax is None)
    
    if create_new_fig:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Obtener valores únicos para el feature
    unique_values = data[feature].unique()
    
    # Asignar colores altamente distintivos para cada grupo
    colors = assign_distinct_category_colors(data, feature)
    
    # Diccionario para almacenar centroides
    centroids = {}
    
    # Graficar puntos por grupo y calcular centroides
    for i, value in enumerate(unique_values):
        subset = data[data[feature] == value]
        x = subset[x_col].values
        y = subset[y_col].values
        
        # Color para este grupo
        color = colors.get(value)
        
        # Graficamos los puntos con mayor tamaño y contraste
        ax.scatter(x, y, alpha=0.8, s=50, label=f"{feature}: {value}", 
                   color=color, edgecolor='black', linewidths=0.5)
        
        # Calculamos el centroide
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        centroids[value] = (centroid_x, centroid_y)
        
        # Graficamos el centroide con mayor prominencia
        ax.scatter(centroid_x, centroid_y, s=200, marker='X', 
                   edgecolor='black', linewidth=1.5, color=color, 
                   label=f"Centroide {value}")
        
        # Agregamos elipse de contorno si se solicita, con color semitransparente
        if show_ellipses and len(x) > 2:
            try:
                # Usamos el mismo color pero con transparencia para la elipse
                ellipse_color = color
                confidence_ellipse(x, y, ax, n_std=2.0, edgecolor=color, 
                                  linewidth=2, linestyle='-', facecolor=ellipse_color)
            except Exception as e:
                print(f"No se pudo dibujar la elipse para {value}: {e}")
    
    # Calculamos las distancias entre centroides
    distances = {}
    for val1, val2 in itertools.combinations(unique_values, 2):
        dist = distance.euclidean(centroids[val1], centroids[val2])
        key = f'{val1} vs {val2}'
        distances[key] = dist
    
    if create_new_fig:
        ax.set_title(f'Clusters por {feature}', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        # Leyenda con mejor formato
        ax.legend(fontsize=10, framealpha=0.7)
        plt.tight_layout()
        return fig, centroids, distances
    
    return centroids, distances

def calculate_statistics(data, feature, x_col, y_col):
    """
    Calcula estadísticas detalladas para los clusters
    """
    unique_values = data[feature].unique()
    stats_df = pd.DataFrame()
    
    # Para cada valor del feature
    for value in unique_values:
        subset = data[data[feature] == value]
        x = subset[x_col].values
        y = subset[y_col].values
        
        # Estadísticas básicas univariadas
        stats_dict = {
            'Grupo': value,
            'Característica': feature,
            'N': len(subset),
            f'{x_col}_Media': subset[x_col].mean(),
            f'{x_col}_Mediana': subset[x_col].median(),
            f'{x_col}_DE': subset[x_col].std(),
            f'{x_col}_Min': subset[x_col].min(),
            f'{x_col}_Max': subset[x_col].max(),
            f'{x_col}_Q1': subset[x_col].quantile(0.25),
            f'{x_col}_Q3': subset[x_col].quantile(0.75),
            f'{y_col}_Media': subset[y_col].mean(),
            f'{y_col}_Mediana': subset[y_col].median(),
            f'{y_col}_DE': subset[y_col].std(),
            f'{y_col}_Min': subset[y_col].min(),
            f'{y_col}_Max': subset[y_col].max(),
            f'{y_col}_Q1': subset[y_col].quantile(0.25),
            f'{y_col}_Q3': subset[y_col].quantile(0.75),
        }
        
        # Correlación entre X e Y
        stats_dict[f'Correlación_{x_col}_{y_col}'] = subset[x_col].corr(subset[y_col])
        
        # Estadísticas bivariadas
        bivariate_stats = calculate_bivariate_measures(x, y)
        
        # Añadir métricas bivariadas al diccionario
        stats_dict['Área_Casco_Convexo'] = bivariate_stats['convex_hull_area']
        stats_dict['Dist_Media_Centroide'] = bivariate_stats['avg_distance_to_centroid']
        stats_dict['Dispersión_Espacial'] = bivariate_stats['spatial_dispersion']
        stats_dict['Dist_Mahalanobis_Media'] = bivariate_stats['mahalanobis_center']
        stats_dict['Dist_Mediana_Espacial'] = bivariate_stats['spatial_median_distance']
        
        # Calcular radio de giro (raíz cuadrada de la dispersión espacial)
        stats_dict['Radio_Giro'] = np.sqrt(bivariate_stats['spatial_dispersion'])
        
        # Densidad aproximada (puntos / área del casco convexo)
        if bivariate_stats['convex_hull_area'] > 0:
            stats_dict['Densidad'] = len(subset) / bivariate_stats['convex_hull_area']
        else:
            stats_dict['Densidad'] = np.nan
        
        # Coeficiente de variación espacial (desviación estándar / media de distancias)
        if bivariate_stats['avg_distance_to_centroid'] > 0:
            distances = [np.linalg.norm([x_i - stats_dict[f'{x_col}_Media'], 
                                        y_i - stats_dict[f'{y_col}_Media']]) 
                        for x_i, y_i in zip(x, y)]
            spatial_cv = np.std(distances) / bivariate_stats['avg_distance_to_centroid']
            stats_dict['Coef_Variación_Espacial'] = spatial_cv
        else:
            stats_dict['Coef_Variación_Espacial'] = np.nan
        
        # Añadir al DataFrame principal
        stats_df = pd.concat([stats_df, pd.DataFrame([stats_dict])], ignore_index=True)
    
    return stats_df

def energy_distance_test(x1, y1, x2, y2, n_permutations=100):  # Reducido a 100 permutaciones
    """
    Versión optimizada de la prueba de distancia de energía
    """
    # Crear arrays de puntos
    points1 = np.column_stack((x1, y1))
    points2 = np.column_stack((x2, y2))
    
    # Usar scipy.spatial.distance para cálculos más eficientes
    from scipy.spatial.distance import pdist, cdist, squareform
    
    def energy_distance_fast(a, b):
        # Distancias dentro de los grupos (mucho más rápido con pdist)
        a_dists = pdist(a)
        b_dists = pdist(b)
        
        # Distancias entre grupos
        ab_dists = cdist(a, b).flatten()
        
        # Calcular estadística de energía
        n_a, n_b = len(a), len(b)
        term1 = np.mean(ab_dists)
        term2 = np.mean(a_dists)
        term3 = np.mean(b_dists)
        
        return 2*term1 - term2 - term3
    
    # Calcular estadística original
    try:
        original_stat = energy_distance_fast(points1, points2)
    except Exception as e:
        print(f"Error calculando estadística original: {e}")
        return np.nan, np.nan
    
    # Limitar el número de permutaciones si hay muchos puntos
    total_points = len(points1) + len(points2)
    if total_points > 100:  # Si hay muchos puntos, reducir permutaciones
        n_permutations = min(n_permutations, 50)
    
    # Permutaciones
    combined = np.vstack((points1, points2))
    n1 = len(points1)
    permuted_stats = []
    
    try:
        for _ in range(n_permutations):
            # Permutación más eficiente
            np.random.shuffle(combined)
            perm1, perm2 = combined[:n1], combined[n1:]
            perm_stat = energy_distance_fast(perm1, perm2)
            permuted_stats.append(perm_stat)
        
        # Calcular p-value
        p_value = np.mean([stat >= original_stat for stat in permuted_stats])
        return original_stat, p_value
    except Exception as e:
        print(f"Error en prueba de permutación: {e}")
        return original_stat, np.nan
    
    return original_distance, p_value

def hotelling_t2_test(x1, y1, x2, y2):
    """
    Realiza la prueba T2 de Hotelling para comparar dos grupos multivariados
    """
    points1 = np.column_stack((x1, y1))
    points2 = np.column_stack((x2, y2))
    
    n1, p1 = points1.shape
    n2, p2 = points2.shape
    
    # Verificar que ambos conjuntos tienen la misma dimensionalidad
    assert p1 == p2, "Los conjuntos de datos deben tener la misma dimensionalidad"
    
    # Medias muestrales
    mean1 = np.mean(points1, axis=0)
    mean2 = np.mean(points2, axis=0)
    
    # Matrices de covarianza muestrales
    cov1 = np.cov(points1, rowvar=False)
    cov2 = np.cov(points2, rowvar=False)
    
    # Matriz de covarianza combinada
    pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)
    
    # Verificar si la matriz de covarianza es invertible
    if np.linalg.det(pooled_cov) <= 0:
        return np.nan, np.nan
    
    try:
        # Calcular la estadística T2
        mean_diff = mean1 - mean2
        inv_pooled_cov = np.linalg.inv(pooled_cov)
        t2 = (n1 * n2) / (n1 + n2) * (mean_diff @ inv_pooled_cov @ mean_diff)
        
        # Calcular F y p-value
        p = p1  # dimensionalidad
        df1 = p
        df2 = n1 + n2 - p - 1
        
        # Convertir T2 a F
        f_stat = ((n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)) * t2
        
        # Calcular el valor p
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        return t2, p_value
    except:
        return np.nan, np.nan

def permutation_manova(x1, y1, x2, y2, n_permutations=1000):
    """
    Realiza una prueba MANOVA basada en permutaciones para datos bivariados
    """
    points1 = np.column_stack((x1, y1))
    points2 = np.column_stack((x2, y2))
    
    combined = np.vstack((points1, points2))
    n1, n2 = len(points1), len(points2)
    ntot = n1 + n2
    
    # Vector de grupo
    group = np.hstack((np.zeros(n1), np.ones(n2)))
    
    # Función para calcular la estadística de Wilks' Lambda
    def wilks_lambda(X, group_vec):
        unique_groups = np.unique(group_vec)
        # Matriz de covarianza total
        total_cov = np.cov(X, rowvar=False)
        
        # Matrices de covarianza dentro de grupos
        within_cov = np.zeros_like(total_cov)
        
        for g in unique_groups:
            group_data = X[group_vec == g]
            n_g = len(group_data)
            group_cov = np.cov(group_data, rowvar=False)
            within_cov += (n_g - 1) * group_cov
        
        # Ajustar para los grados de libertad
        within_cov /= (ntot - len(unique_groups))
        
        # Calcular Lambda de Wilks
        try:
            lambda_val = np.linalg.det(within_cov) / np.linalg.det(total_cov)
            return lambda_val
        except:
            return np.nan
    
    # Calcular estadística original
    original_lambda = wilks_lambda(combined, group)
    
    # Realizar permutaciones
    permuted_lambdas = []
    for _ in range(n_permutations):
        # Mezclar las etiquetas de grupo
        permuted_group = np.random.permutation(group)
        # Calcular estadística permutada
        perm_lambda = wilks_lambda(combined, permuted_group)
        permuted_lambdas.append(perm_lambda)
    
    # Calcular valor p (proporción de permutaciones con lambda menor o igual al original)
    p_value = np.mean([l <= original_lambda for l in permuted_lambdas if not np.isnan(l)])
    
    return original_lambda, p_value

def statistical_tests(data, feature, x_col, y_col):
    """
    Realiza pruebas estadísticas entre grupos con inferencia estadística completa
    """
    unique_values = sorted(data[feature].unique())
    test_results = []
    
    # Si hay solo un grupo, no podemos hacer pruebas comparativas
    if len(unique_values) <= 1:
        return pd.DataFrame()
    
    # Para cada par de grupos
    for val1, val2 in itertools.combinations(unique_values, 2):
        group1 = data[data[feature] == val1]
        group2 = data[data[feature] == val2]
        
        # Valores de X e Y para ambos grupos
        x1 = group1[x_col].values
        y1 = group1[y_col].values
        x2 = group2[x_col].values
        y2 = group2[y_col].values
        
        # Verificar que hay suficientes datos
        if min(len(x1), len(x2)) < 3:
            continue
        
        # t-test para la coordenada X
        t_stat_x, p_value_x = stats.ttest_ind(x1, x2, equal_var=False)
        
        # t-test para la coordenada Y
        t_stat_y, p_value_y = stats.ttest_ind(y1, y2, equal_var=False)
        
        # Mann-Whitney U test (no paramétrico) para la coordenada X
        try:
            u_stat_x, p_value_ux = stats.mannwhitneyu(x1, x2, alternative='two-sided')
        except:
            u_stat_x, p_value_ux = np.nan, np.nan
            
        # Mann-Whitney U test para la coordenada Y
        try:
            u_stat_y, p_value_uy = stats.mannwhitneyu(y1, y2, alternative='two-sided')
        except:
            u_stat_y, p_value_uy = np.nan, np.nan
        
        # Prueba multivariada: Hotelling's T2
        t2_stat, t2_pvalue = hotelling_t2_test(x1, y1, x2, y2)
        
        # Distancia de energía con prueba de permutación
        energy_dist, energy_pvalue = energy_distance_test(x1, y1, x2, y2)
        
        # MANOVA basado en permutaciones
        manova_stat, manova_pvalue = permutation_manova(x1, y1, x2, y2)
        
        # Distancia entre centroides
        centroid1 = np.mean(np.column_stack((x1, y1)), axis=0)
        centroid2 = np.mean(np.column_stack((x2, y2)), axis=0)
        centroid_distance = distance.euclidean(centroid1, centroid2)
        
        # Calcular solapamiento de distribuciones usando distancia de Bhattacharyya
        try:
            # Matrices de covarianza
            cov1 = np.cov(np.column_stack((x1, y1)), rowvar=False)
            cov2 = np.cov(np.column_stack((x2, y2)), rowvar=False)
            
            # Matriz de covarianza promedio
            cov_avg = (cov1 + cov2) / 2
            
            # Verificar que las matrices son invertibles
            if np.linalg.det(cov_avg) > 0:
                # Distancia de Bhattacharyya
                term1 = 0.125 * ((centroid1 - centroid2).T @ np.linalg.inv(cov_avg) @ (centroid1 - centroid2))
                
                if np.linalg.det(cov1) > 0 and np.linalg.det(cov2) > 0:
                    term2 = 0.5 * np.log(np.linalg.det(cov_avg) / 
                                        np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
                    
                    bhattacharyya_dist = term1 + term2
                    overlap = np.exp(-bhattacharyya_dist)
                else:
                    overlap = np.nan
            else:
                overlap = np.nan
        except:
            overlap = np.nan
        
        test_results.append({
            'Comparación': f'{val1} vs {val2}',
            'Característica': feature,
            f't-test {x_col} t-stat': t_stat_x,
            f't-test {x_col} p-value': p_value_x,
            f't-test {y_col} t-stat': t_stat_y,
            f't-test {y_col} p-value': p_value_y,
            f'Mann-Whitney {x_col} U-stat': u_stat_x,
            f'Mann-Whitney {x_col} p-value': p_value_ux,
            f'Mann-Whitney {y_col} U-stat': u_stat_y,
            f'Mann-Whitney {y_col} p-value': p_value_uy,
            'Hotelling T2 estadístico': t2_stat,
            'Hotelling T2 p-value': t2_pvalue,
            'Distancia_Energía': energy_dist,
            'Distancia_Energía p-value': energy_pvalue,
            'MANOVA Wilks Lambda': manova_stat,
            'MANOVA p-value': manova_pvalue,
            'Distancia_Centroides': centroid_distance,
            'Coeficiente_Solapamiento': overlap
        })
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(test_results)
    
    # Ajustar los valores p para comparaciones múltiples si hay suficientes tests
    if len(results_df) > 1:
        # Columnas que contienen valores p
        p_value_cols = [col for col in results_df.columns if 'p-value' in col.lower()]
        
        for col in p_value_cols:
            # Obtener valores p que no son NaN
            valid_pvals = results_df[col].dropna().values
            
            if len(valid_pvals) > 1:  # Necesitamos al menos 2 valores para la corrección
                # Aplicar corrección FDR (Método de Benjamini-Hochberg)
                try:
                    _, adjusted_pvals, _, _ = multipletests(valid_pvals, method='fdr_bh')
                    
                    # Crear una nueva columna con los valores p ajustados
                    adjusted_col = f"{col} (ajustado)"
                    results_df[adjusted_col] = np.nan
                    
                    # Asignar los valores p ajustados a las filas correspondientes
                    idx = results_df[col].notna()
                    results_df.loc[idx, adjusted_col] = adjusted_pvals
                except:
                    print(f"No se pudo aplicar corrección para {col}")
    
    return results_df

def distance_dataframe(centroids_dict, distances_dict, feature):
    """
    Crea un DataFrame con la información de centroides y distancias
    """
    # DataFrame para centroides
    centroid_rows = []
    for group, (x, y) in centroids_dict.items():
        centroid_rows.append({
            'Grupo': group,
            'Característica': feature,
            'Centroide_X': x,
            'Centroide_Y': y
        })
    
    centroid_df = pd.DataFrame(centroid_rows)
    
    # DataFrame para distancias con más detalle
    distance_rows = []
    for comparison, dist in distances_dict.items():
        # Separar los grupos en la comparación
        groups = comparison.split(' vs ')
        if len(groups) == 2:
            group1, group2 = groups
            # Agregar coordenadas de los centroides a la información de distancias
            x1, y1 = centroids_dict[group1]
            x2, y2 = centroids_dict[group2]
            
            distance_rows.append({
                'Comparación': comparison,
                'Característica': feature,
                'Grupo1': group1,
                'Grupo2': group2,
                'Centroide1_X': x1,
                'Centroide1_Y': y1,
                'Centroide2_X': x2,
                'Centroide2_Y': y2,
                'Distancia': dist,
                'Dist_X': abs(x1 - x2),
                'Dist_Y': abs(y1 - y2)
            })
    
    distance_df = pd.DataFrame(distance_rows)
    
    return centroid_df, distance_df

def select_columns_dialog(data):
    """
    Crea un diálogo para seleccionar columnas para el análisis
    """
    root = tk.Tk()
    root.withdraw()
    
    # Mostrar columnas disponibles
    columns_str = "\n".join([f"{i+1}. {col}" for i, col in enumerate(data.columns)])
    messagebox.showinfo("Columnas disponibles", f"Columnas disponibles:\n\n{columns_str}")
    
    # Solicitar columna X
    x_col = None
    while x_col is None:
        x_response = simpledialog.askinteger("Selección de columna X", 
                                        "Ingresa el número de la columna para el eje X (coordenadas):",
                                        minvalue=1, maxvalue=len(data.columns))
        if x_response:
            x_col = data.columns[x_response-1]
        else:
            messagebox.showwarning("Entrada inválida", "Por favor, selecciona una columna válida.")
    
    # Solicitar columna Y
    y_col = None
    while y_col is None:
        y_response = simpledialog.askinteger("Selección de columna Y", 
                                        "Ingresa el número de la columna para el eje Y (coordenadas):",
                                        minvalue=1, maxvalue=len(data.columns))
        if y_response:
            y_col = data.columns[y_response-1]
        else:
            messagebox.showwarning("Entrada inválida", "Por favor, selecciona una columna válida.")
    
    # Solicitar columnas para agrupar (múltiples)
    group_cols = []
    
    # Primero mostrar las columnas otra vez para refrescar la memoria
    messagebox.showinfo("Columnas para agrupar", 
                        f"Columnas disponibles para agrupar:\n\n{columns_str}\n\nPuedes seleccionar múltiples.")
    
    while True:
        group_response = simpledialog.askinteger("Selección de columna para agrupar", 
                                            "Ingresa el número de una columna para agrupar (sexo, edad, etc.)\n"
                                            "o ingresa 0 para terminar la selección:",
                                            minvalue=0, maxvalue=len(data.columns))
        
        if group_response == 0 or group_response is None:
            break
        
        group_col = data.columns[group_response-1]
        if group_col not in group_cols:
            group_cols.append(group_col)
    
    root.destroy()
    
    if not group_cols:
        messagebox.showwarning("Advertencia", "No seleccionaste columnas para agrupar. Se usará la primera columna categórica.")
        # Intenta encontrar una columna categórica
        for col in data.columns:
            if data[col].dtype == 'object' or len(data[col].unique()) < 10:
                group_cols.append(col)
                break
        
        if not group_cols:
            group_cols = [data.columns[0]]  # Última opción
    
    return x_col, y_col, group_cols

def create_output_path(input_file_path):
    """
    Solicita al usuario una carpeta para guardar los resultados
    """
    root = tk.Tk()
    root.withdraw()
    
    file_name = os.path.basename(input_file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    
    # Solicitar directorio para guardar resultados
    messagebox.showinfo("Carpeta de destino", 
                       "A continuación, selecciona la carpeta donde quieres guardar los resultados")
    
    output_dir = filedialog.askdirectory(
        title="Selecciona la carpeta para guardar los resultados"
    )
    
    # Si no se seleccionó ninguna carpeta, usar el directorio del archivo original
    if not output_dir:
        base_dir = os.path.dirname(input_file_path)
        output_dir = os.path.join(base_dir, f"{file_name_no_ext}_resultados")
        messagebox.showinfo("Información", 
                           f"No se seleccionó carpeta. Se usará: {output_dir}")
    
    # Crear la carpeta si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir
    
def ask_for_superposition():
    """
    Pregunta al usuario si desea superponer los análisis
    """
    root = tk.Tk()
    root.withdraw()
    response = messagebox.askyesno("Superposición", "¿Deseas superponer los análisis en un mismo gráfico?")
    root.destroy()
    return response

def ask_for_ellipses():
    """
    Pregunta al usuario si desea mostrar elipses de confianza
    """
    root = tk.Tk()
    root.withdraw()
    response = messagebox.askyesno("Elipses de confianza", "¿Deseas mostrar elipses de confianza en los gráficos?")
    root.destroy()
    return response

def main():
    print("=== Análisis de Clusters Avanzado en Datos ===")
    
    # Solicitar al usuario que seleccione el archivo
    file_path = select_file()
    if not file_path:
        print("No se seleccionó ningún archivo. Saliendo...")
        return
    
    print(f"Archivo seleccionado: {file_path}")
    
    # Cargar los datos
    data = load_data(file_path)
    if data is None:
        return
    
    print(f"Datos cargados con éxito. Shape: {data.shape}")
    
    # Mostrar las primeras filas para verificar
    print("\nPrimeras filas de los datos:")
    print(data.head())
    
    # Seleccionar columnas para el análisis
    x_col, y_col, group_cols = select_columns_dialog(data)
    
    print(f"\nAnalizando clusters usando:")
    print(f"- Eje X: {x_col}")
    print(f"- Eje Y: {y_col}")
    print(f"- Agrupando por: {', '.join(group_cols)}")
    
    # Preguntar por superposición
    superpose = ask_for_superposition()
    
    # Preguntar por elipses
    show_ellipses = ask_for_ellipses()
    
    # Crear la ruta de salida
    output_dir = create_output_path(file_path)
    print(f"\nLos resultados se guardarán en: {output_dir}")
    
    # Crear listas para almacenar DataFrames
    all_stats = []
    all_tests = []
    all_centroids = []
    all_distances = []
    
     # Si superponemos, creamos una sola figura para todos los análisis
    if superpose:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title(f'Análisis de clusters superpuestos\n({", ".join(group_cols)})')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    
    # Para cada columna de agrupación
    for group_col in group_cols:
        print(f"\nAnalizando clusters por: {group_col}")
        
        if superpose:
            # Añadir a la figura existente
            centroids, distances = analyze_clusters(data, group_col, x_col, y_col, ax=ax, show_ellipses=show_ellipses)
        else:
            # Crear una nueva figura
            fig, centroids, distances = analyze_clusters(data, group_col, x_col, y_col, show_ellipses=show_ellipses)
            
            # Guardar la figura individual
            output_file = os.path.join(output_dir, f"clusters_por_{group_col}.png")
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Figura guardada como: {output_file}")
        
        # Calcular estadísticas
        stats_df = calculate_statistics(data, group_col, x_col, y_col)
        all_stats.append(stats_df)
        
        # Realizar pruebas estadísticas
        tests_df = statistical_tests(data, group_col, x_col, y_col)
        if not tests_df.empty:
            all_tests.append(tests_df)
        
        # Convertir centroides y distancias a DataFrames
        centroid_df, distance_df = distance_dataframe(centroids, distances, group_col)
        all_centroids.append(centroid_df)
        all_distances.append(distance_df)
        
        # Mostrar distancias
        print("\nDistancias entre centroides:")
        for key, value in distances.items():
            print(f"{key}: {value:.4f}")
    
    # Si estamos superponiendo, guardar la figura combinada
    if superpose:
        # Ajustar la leyenda para evitar solapamientos
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
        
        output_file = os.path.join(output_dir, "clusters_superpuestos.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nFigura superpuesta guardada como: {output_file}")
    
    # Combinar todos los DataFrames
    combined_stats = pd.concat(all_stats, ignore_index=True)
    combined_centroids = pd.concat(all_centroids, ignore_index=True)
    combined_distances = pd.concat(all_distances, ignore_index=True)
    
    # Combinar tests si existen
    if all_tests:
        combined_tests = pd.concat(all_tests, ignore_index=True)
    else:
        combined_tests = pd.DataFrame({"Mensaje": ["No hay suficientes grupos para realizar pruebas estadísticas"]})
    
    # Crear datos para el pivot table de distancias
    pivot_distances = []
    
    for group_col in group_cols:
        subset = combined_distances[combined_distances['Característica'] == group_col]
        for _, row in subset.iterrows():
            pivot_distances.append({
                'Comparación': row['Comparación'],
                'Característica': row['Característica'],
                'Distancia': row['Distancia']
            })
    
    pivot_df = pd.DataFrame(pivot_distances)
    
    # Guardar resultados en Excel
    excel_file = os.path.join(output_dir, "resultados_analisis_clusters.xlsx")
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Hoja principal con un resumen
        pd.DataFrame({
            'Análisis': ['Clusters en datos'],
            'Archivo original': [os.path.basename(file_path)],
            'Columna X': [x_col],
            'Columna Y': [y_col],
            'Variables de agrupación': [', '.join(group_cols)],
            'Fecha de análisis': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        }).to_excel(writer, sheet_name='Resumen', index=False)
        
        # Estadísticas detalladas
        combined_stats.to_excel(writer, sheet_name='Estadísticas', index=False)
        
        # Pruebas estadísticas
        combined_tests.to_excel(writer, sheet_name='Pruebas Estadísticas', index=False)
        
        # Información de centroides
        combined_centroids.to_excel(writer, sheet_name='Centroides', index=False)
        
        # Información detallada de distancias
        combined_distances.to_excel(writer, sheet_name='Distancias', index=False)
        
        # Pivot table de distancias para mejor visualización
        if not pivot_df.empty:
            pivot_dist = pivot_df.pivot_table(
                index='Comparación', 
                columns='Característica', 
                values='Distancia',
                aggfunc='first'
            ).reset_index()
            pivot_dist.to_excel(writer, sheet_name='Matriz de Distancias', index=False)
        
        # También guardar los datos originales
        data.to_excel(writer, sheet_name='Datos Originales', index=False)
        
        # Estadísticas adicionales por grupo
        for group_col in group_cols:
            group_stats = combined_stats[combined_stats['Característica'] == group_col]
            group_stats.to_excel(writer, sheet_name=f'Stats_{group_col[:28]}', index=False)
    
    print(f"\nResultados guardados en Excel: {excel_file}")
    
    # Guardar gráficos de distancias
    if not combined_distances.empty:
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        
        # Crear gráfico de barras para las distancias
        sns.barplot(data=pivot_df, x='Comparación', y='Distancia', hue='Característica', ax=ax_dist)
        ax_dist.set_title('Distancias entre centroides')
        ax_dist.set_ylabel('Distancia euclidiana')
        ax_dist.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        # Guardar gráfico de distancias
        distance_file = os.path.join(output_dir, "distancias_entre_centroides.png")
        fig_dist.savefig(distance_file, dpi=300, bbox_inches='tight')
        print(f"Gráfico de distancias guardado como: {distance_file}")
    
    # Mostrar todas las figuras
    plt.show()
    
    print("\n=== Análisis completado ===")
    print(f"Todos los resultados se han guardado en: {output_dir}")

if __name__ == "__main__":
    # Configuración visual
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
    
    main()