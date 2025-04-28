import os
import re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import threading
import queue
import sys
import traceback
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='cell_data_filter.log',
    filemode='w'
)

class CellDataFilterApp:
    def __init__(self, master):
        """Initialize the application"""
        self.master = master
        master.title("Cell Data Interactive Filter")
        master.geometry("900x1000")

        # Thread-safe queue for progress updates
        self.progress_queue = queue.Queue()

        # Configure logging
        logging.info("Initializing Cell Data Filter Application")

        # Initial setup
        self.initialize_data()
        self.create_widgets()

    def initialize_data(self):
        """Initialize all data sources with robust error handling"""
        try:
            # Attempt to import ABC Atlas modules
            try:
                from abc_atlas_access.abc_atlas_cache.anndata_utils import get_gene_data
                from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
                self.get_gene_data = get_gene_data
                self.AbcProjectCache = AbcProjectCache
            except ImportError as e:
                logging.error(f"Failed to import ABC Atlas modules: {e}")
                messagebox.showerror("Import Error", f"Required modules not found: {e}")
                sys.exit(1)

            # Configure cache
            self.download_base = Path('../../data/abc_atlas')
            
            # Log cache directory details
            logging.info(f"Cache base directory: {self.download_base}")
            logging.info(f"Cache directory exists: {os.path.exists(self.download_base)}")
            
            # Initialize cache
            self.abc_cache = self.AbcProjectCache.from_cache_dir(self.download_base)

            # Load metadata
            self.load_metadata()

        except Exception as e:
            logging.error(f"Data Initialization Error: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to initialize: {str(e)}")
            self.master.quit()

    def load_metadata(self):
        """Load metadata from ABC Atlas cache"""
        try:
            logging.info("Starting metadata loading")
            start_time = time.time()

            # Cell metadata
            self.cell_metadata = self.abc_cache.get_metadata_dataframe(
                directory='Zeng-Aging-Mouse-10Xv3',
                file_name='cell_metadata',
                dtype={'cell_label': str, 'wmb_cluster_alias': 'Int64'}
            )
            
            # Cell cluster annotations
            self.cell_cluster_annotations = self.abc_cache.get_metadata_dataframe(
                directory='Zeng-Aging-Mouse-10Xv3',
                file_name='cell_cluster_annotations'
            )
            
            # Gene metadata
            self.gene = self.abc_cache.get_metadata_dataframe(
                directory='WMB-10X', 
                file_name='gene'
            ).set_index('gene_identifier')

            # Sorted gene list
            self.gene_list = sorted(self.gene['gene_symbol'].unique())

            logging.info(f"Metadata loading completed in {time.time() - start_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Metadata Loading Error: {str(e)}")
            messagebox.showerror("Metadata Loading Error", f"Failed to load metadata: {str(e)}")
            self.master.quit()

    def create_widgets(self):
        """Create application UI components"""
        # Main Frame
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Filtering Frame
        filter_frame = ttk.LabelFrame(main_frame, text="Filtering Options")
        filter_frame.pack(fill='x', padx=10, pady=10)

        # Region of Interest Dropdown
        ttk.Label(filter_frame, text="Region of Interest:").pack(anchor='w')
        self.roi_var = tk.StringVar()
        roi_dropdown = ttk.Combobox(
            filter_frame, 
            textvariable=self.roi_var, 
            values=[''] + sorted(self.cell_metadata['region_of_interest_label'].unique().tolist())
        )
        roi_dropdown.pack(fill='x', padx=5, pady=5)

        # Neurotransmitter Dropdown
        ttk.Label(filter_frame, text="Neurotransmitter:").pack(anchor='w')
        self.neurotransmitter_var = tk.StringVar()
        neurotransmitter_dropdown = ttk.Combobox(
            filter_frame, 
            textvariable=self.neurotransmitter_var, 
            values=[''] + sorted(self.cell_cluster_annotations['neurotransmitter_combined_label'].unique().tolist())
        )
        neurotransmitter_dropdown.pack(fill='x', padx=5, pady=5)

        # Gene Selection Frame
        gene_frame = ttk.LabelFrame(main_frame, text="Gene Selection")
        gene_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Gene Search Entry
        search_frame = ttk.Frame(gene_frame)
        search_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(search_frame, text="Search Genes:").pack(side='left')
        self.gene_search_var = tk.StringVar()
        self.gene_search_var.trace('w', self.update_gene_list)
        gene_search_entry = ttk.Entry(search_frame, textvariable=self.gene_search_var, width=50)
        gene_search_entry.pack(side='left', expand=True, fill='x', padx=5)

        # Select All/None Buttons
        button_frame = ttk.Frame(gene_frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(button_frame, text="Select All", command=self.select_all_genes).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Deselect All", command=self.deselect_all_genes).pack(side='left', padx=5)

        # Gene Listbox
        self.gene_listbox = tk.Listbox(gene_frame, selectmode=tk.MULTIPLE, height=15)
        self.gene_listbox.pack(fill='both', expand=True, padx=5, pady=5)

        # Progress Frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill='x', padx=10, pady=10)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            orient='horizontal', 
            length=750, 
            mode='determinate'
        )
        self.progress_bar.pack(fill='x', padx=5, pady=5)

        # Progress Label
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(padx=5, pady=5)

        # Filter Button
        self.filter_button = ttk.Button(main_frame, text="Apply Filters", command=self.start_filter_thread)
        self.filter_button.pack(fill='x', padx=10, pady=10)

        # Populate gene listbox 
        self.update_gene_list()
        self.select_all_genes()

    def update_gene_list(self, *args):
        """Update gene listbox based on search term"""
        search_term = self.gene_search_var.get().lower()
        
        self.gene_listbox.delete(0, tk.END)
        
        for gene in self.gene_list:
            if search_term == '' or search_term in gene.lower():
                self.gene_listbox.insert(tk.END, gene)

    def select_all_genes(self):
        """Select all genes in the current listbox"""
        self.gene_listbox.selection_clear(0, tk.END)
        self.gene_listbox.selection_set(0, tk.END)

    def deselect_all_genes(self):
        """Deselect all genes in the current listbox"""
        self.gene_listbox.selection_clear(0, tk.END)

    def start_filter_thread(self):
        """Start filtering in a separate thread"""
        # Reset progress
        self.progress_bar['value'] = 0
        self.progress_label['text'] = "Starting processing..."

        # Disable filter button during processing
        self.filter_button.config(state='disabled')

        # Start thread
        thread = threading.Thread(target=self.apply_filters, daemon=True)
        thread.start()

        # Start progress checking
        self.check_progress()

    def check_progress(self):
        """Check for progress updates from the thread"""
        try:
            try:
                msg = self.progress_queue.get_nowait()
                
                if isinstance(msg, tuple):
                    progress, text = msg
                    self.progress_bar['value'] = progress
                    self.progress_label['text'] = text
                elif msg == 'complete':
                    self.progress_bar['value'] = 100
                    self.progress_label['text'] = "Processing Complete"
                    self.filter_button.config(state='normal')
                    return
            except queue.Empty:
                pass
            
            if self.progress_bar['value'] < 100:
                self.master.after(100, self.check_progress)
        
        except Exception as e:
            logging.error(f"Progress check error: {e}")
            messagebox.showerror("Error", str(e))

    def apply_filters(self):
        """Apply selected filters to the dataset"""
        try:
            logging.info("Starting filter process")
            self.progress_queue.put((10, "Initializing filter process..."))

            # Get selected filters
            selected_roi = self.roi_var.get()
            selected_neurotransmitter = self.neurotransmitter_var.get()
            selected_genes = [self.gene_listbox.get(idx) for idx in self.gene_listbox.curselection()]

            # Merge cell metadata with cluster annotations
            merged_cells = pd.merge(
                self.cell_metadata,
                self.cell_cluster_annotations[['cell_label', 'neurotransmitter_combined_label', 'cluster_name']],
                on='cell_label',
                how='inner'
            )

            # Filter merged cells
            filtered_cells = merged_cells.copy()

            # Apply ROI filter
            if selected_roi:
                self.progress_queue.put((30, f"Filtering by Region: {selected_roi}"))
                filtered_cells = filtered_cells[
                    filtered_cells['region_of_interest_label'] == selected_roi
                ]

            # Apply Neurotransmitter filter
            if selected_neurotransmitter:
                self.progress_queue.put((50, f"Filtering by Neurotransmitter: {selected_neurotransmitter}"))
                filtered_cells = filtered_cells[
                    filtered_cells['neurotransmitter_combined_label'] == selected_neurotransmitter
                ]

            # Set index for cell filtering
            filtered_cells = filtered_cells.set_index('cell_label')

            # Optimized Gene Filtering
            if selected_genes:
                self.progress_queue.put((70, f"Processing {len(selected_genes)} genes"))
                
                try:
                    # Single call to get gene data for all selected genes
                    gene_data = self.get_gene_data(
                        abc_atlas_cache=self.abc_cache,
                        all_cells=filtered_cells,
                        all_genes=self.gene,
                        selected_genes=selected_genes,
                        data_type="log2"
                    )
                    
                    # Check if gene data is not empty
                    if not gene_data.empty:
                        # Intersect and merge
                        valid_cells = gene_data.index.intersection(filtered_cells.index)
                        valid_gene_data = gene_data.loc[valid_cells]
                        final_cells_with_genes = filtered_cells.loc[valid_cells].merge(
                            valid_gene_data, 
                            left_index=True, 
                            right_index=True
                        )
                    else:
                        raise ValueError("No gene data retrieved")
                
                except Exception as error:
                    logging.error(f"Gene data retrieval error: {error}")
                    self.progress_queue.put((0, "Error processing genes"))
                    
                    self.master.after(0, lambda msg=str(error): messagebox.showerror(
                        "Processing Error", 
                        f"Could not retrieve gene data: {msg}"
                    ))
                    return
            else:
                final_cells_with_genes = filtered_cells

            # Lista de columnas a eliminar
            columns_to_remove = [
                "cell_barcode", "gene_count", "umi_count", "doublet_score", "x", "y", 
                "cluster_alias", "cell_in_wmb_study", "wmb_cluster_alias", "library_label", 
                "alignment_job_id", "library_method", "barcoded_cell_sample_label", 
                "enrichment_population", "library_in_wmb_study", "donor_label", 
                "population_sampling", "donor_genotype", "donor_age", "donor_in_wmb_study", 
                "feature_matrix_label", "dataset_label", "abc_sample_id", 
            ]

            # Limpiar columnas, ignorando las que no existen
            final_cells_with_genes_cleaned = final_cells_with_genes.drop(columns=columns_to_remove, errors='ignore')
            
            final_cells_with_genes_cleaned['cluster_name'] = final_cells_with_genes_cleaned['cluster_name'].apply(
            lambda x: re.sub(r'^\d+_|_\d+$', '', x)
            )

            # Ask user to select output directory
            output_directory = filedialog.askdirectory(title="Select Directory to Save Files")
            
            # Verify directory was selected
            if not output_directory:
                self.progress_queue.put((0, "Operation cancelled by user"))
                messagebox.showinfo("Operation Cancelled", "No directory selected. Export aborted.")
                return

            # Prepare output files
            output_prefix = f"{selected_roi}_{selected_neurotransmitter}_" if selected_roi or selected_neurotransmitter else ""
            output_prefix += f"{len(selected_genes)}_genes"

            # Save full dataset
            full_output_path = os.path.join(output_directory, f"{output_prefix}_full.csv")
            final_cells_with_genes.to_csv(full_output_path, index=True)
            logging.info(f"Full dataset saved to {full_output_path}")

            # Save cleaned dataset
            cleaned_output_path = os.path.join(output_directory, f"{output_prefix}_cleaned.csv")
            final_cells_with_genes_cleaned.to_csv(cleaned_output_path, index=True)
            logging.info(f"Cleaned dataset saved to {cleaned_output_path}")

            # Show success message
            self.master.after(0, lambda: messagebox.showinfo(
                "Export Successful", 
                f"Datasets saved in:\n{output_directory}\n"
                f"Full dataset: {os.path.basename(full_output_path)}\n"
                f"Cleaned dataset: {os.path.basename(cleaned_output_path)}\n"
                f"Total Genes Processed: {len(selected_genes)}"
            ))

            # Mark processing as complete
            self.progress_queue.put('complete')

        except Exception as e:
            error_msg = f"Filter Process Error: {str(e)}"
            logging.error(error_msg)
            traceback.print_exc()
            
            self.progress_queue.put((0, error_msg))
            
            # Show error on main thread
            self.master.after(0, lambda msg=error_msg: messagebox.showerror("Processing Error", msg))

def main():
    """Main entry point of the application"""
    logging.info("Starting Cell Data Interactive Filter")
    print("Logging initialized. Check cell_data_filter.log for detailed information.")
    
    root = tk.Tk()
    app = CellDataFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()