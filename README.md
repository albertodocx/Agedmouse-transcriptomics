<img width="612" alt="Imagen 1" src="https://github.com/user-attachments/assets/3cde9cb2-cc53-4541-b029-1218a7cf1b54" />

# Aged Mouse Transcriptomic Analysis

This repository provides scripts and a visualization app for analyzing transcriptomic data from aging mice. The workflow includes data extraction, gene filtering, comparative analysis, and visual inspection using UMAP projections.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Extraction](#1-data-extraction)
  - [2. Visualization](#2-visualization)
  - [3. Comparative Analysis](#3-comparative-analysis)
  - [4. Outlier Detection](#4-outlier-detection)
- [Output Files](#output-files)
- [Example Commands](#example-commands)

## Installation

1. Ensure you have Anaconda installed.

2. Install the required Python library from the Allen Brain Cell Atlas:

```bash
(base) your-disk:\your-path> git clone https://github.com/alleninstitute/abc_atlas_access.git
```

More details can be found in the official [Get Started tutorial](https://alleninstitute.github.io/abc_atlas_access/notebooks/getting_started.html).

3. Open a terminal (cmd) from the Anaconda environment and start Jupyter Notebook:

```bash
(base) C:\Users\YourUser> jupyter notebook
```

4. Open a new terminal within Jupyter by selecting:

```
New > Python [conda env:base]
```

5. Restart the kernel via:

```
Kernel > Restart Kernel
```

6. Import `ABCprojectcache` as instructed in the tutorial.

## Usage

### 1. Data Extraction

Run `agedmousetranscr.py` to extract and filter cell and gene data.

- **Input:** Data and metadata from the cache
- **Output:**
  - `complete_alldata.csv`: All expression data and metadata
  - `cleaned_alldata.csv`: Filtered expression data with key metadata
  - `umap_data.xlsx`: UMAP coordinates for visualization

Customize the script to select desired cells and genes by editing the corresponding filters.

### 2. Visualization

Run `agedmouseapp.py` to visualize the UMAP projection.

- **Input:** `umap_data.xlsx`
- **Features:**
  - Color by cell type, sex, or age
  - Filter by different characteristics
  - Export UMAP figures as `.svg` files

### 3. Comparative Analysis

Run `comparative_analysis.py` to generate gene expression comparisons between two groups.

- **Input:** `Cleaned_alldata.csv`
- **Output:** `resultado_characteristic_genescomunes_group1_group2.csv`

The output contains mean gene expressions for each group, facilitating further analysis.

### 4. Outlier Detection

Run `outliers_analysis.py` to identify differential gene expression outliers.

- **Input:** `resultado_characteristic_genescomunes_group1_group2.csv`
- **Output:** `outlier_analysis.csv`

The script calculates expression differences and classifies genes as:
- Normal (within Q1 and Q3 quartiles)
- Outlier IQR (outside Q1 and Q3 quartiles)
- Extreme (above +5std or below -5std)

Additionally, boxplots and heatmaps are generated for better visualization.

## Output Files

- `Complete_alldata.csv`: Full dataset with all metadata
- `Cleaned_alldata.csv`: Filtered dataset with essential metadata
- `Umap_data.xlsx`: UMAP projection data
- `resultado_characteristic_genescomunes_group1_group2.csv`: Comparative gene expression
- `outlier_analysis.csv`: Outlier detection and classification

## Example Commands

1. Run a Python script:

```bash
(base) your-disk:\your-path> python agedmousetranscr.py
```

2. Start the visualization app using Streamlit:

```bash
(base) your-disk:\your-path> streamlit run agedmouseapp.py
```

![Picture1](https://github.com/user-attachments/assets/2e7933dc-1370-44cf-bcd5-471ec7eefe9d)


---

By following these steps, you can perform detailed transcriptomic analyses and identify key gene expression differences in aging mice. Happy analyzing! 🎉

