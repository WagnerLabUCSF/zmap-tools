

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt



# DATA PREPROCESSING & QUALITY FILTERING



def get_sampling_stats(adata, groupby=''):

    groups = np.unique(adata.obs[groupby])

    if not set(['total_counts','n_genes_by_counts']).issubset(adata.obs.columns):
        'Calculating QC metrics'
        sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    lib_umi_per_cell = []
    lib_genes_per_cell = []
    for group in groups:
      lib_umi_per_cell.append(np.mean(adata.obs['total_counts'][adata.obs[groupby]==group]))
      lib_genes_per_cell.append(np.mean(adata.obs['n_genes_by_counts'][adata.obs[groupby]==group]))
      
    df = pd.DataFrame(data={'UMI per Cell': lib_umi_per_cell, 'Genes per Cell': lib_genes_per_cell}, index=groups)
    return df


def filter_abundant_barcodes(adata, filter_cells=False, threshold=1000, upper_threshold=np.inf, library_id='', save_path='./figures/'):
    '''
    Plots a weighted histogram of transcripts per cell barcode for guiding the
    placement of a filtering threshold. Returns a filtered version of adata.  
    '''

    # if necessary, create the output directory
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # use adata.uns['library_id'] if it exists
    if not library_id:
      if 'library_id' in adata.uns:
        library_id = adata.uns['library_id']

    # Sum total UMI counts and genes for each cell-barcode, save to obs
    counts = np.array(adata.X.sum(1))
    genes = np.array(adata.X.astype(bool).sum(axis=1))
    adata.obs['total_counts'] = counts
    adata.obs['n_genes_by_counts'] = genes
    ix = np.where((counts > threshold) & (counts < upper_threshold), True, False)

    # Plot and format a weighted cell-barcode counts histogram
    sc.set_figure_params(dpi=100, figsize=[4,4], fontsize=12)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(counts, bins=np.logspace(0, 6, 100), weights=counts / sum(counts))
    ax.set_xscale('log')
    ax.set_xlabel('Transcripts per cell barcode')
    ax.set_ylabel('Fraction of total transcripts')
    ax.set_title(library_id)
    ax.text(0.99,0.95, str(np.sum(ix)) + '/' + str(counts.shape[0]) + ' cells retained', ha='right', va='center', transform=ax.transAxes)

    # Overlay the counts thresholds as vertical lines
    ax.plot([upper_threshold, upper_threshold], [0, ax.get_ylim()[1]])
    ax.plot([threshold, threshold], [0, ax.get_ylim()[1]])

    # Save figure to file
    fig.tight_layout()
    plt.savefig(save_path + 'barcode_hist_' + library_id + '.png')
    plt.show()
    plt.close()

    # Print the number of cell barcodes that will be retained 
    print('Barcode Filtering ' + library_id + ' (' + str(np.sum(ix)) + '/' + str(counts.shape[0]) + ' cells retained)')
    print()

    # If requested, return a filtered version of adata
    if filter_cells:
        sc.pp.filter_cells(adata, min_counts=threshold, inplace=True)
        return adata


def filter_mito(adata, filter_cells=False, upper_threshold=100, lower_threshold=0, library_id='', save_path='./figures/'):
    '''
    Plots a weighted histogram of % mitochondrial transcripts per cell barcode for guiding the
    placement of filtering thresholds. Returns a filtered version of adata if filter_cells=True.  
    '''

    # If necessary, create the output directory
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    # Use adata.uns['library_id'] if it exists
    if not library_id:
      if 'library_id' in adata.uns:
        library_id = adata.uns['library_id']

    # Calculate QC metric for % mitochondrial counts per cell
    adata.var["mito"] = adata.var_names.str.startswith(('mt-','MT-'))
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS','rps','RPL','rpl'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mito','ribo'], inplace=True)
    counts = adata.obs['pct_counts_mito']
    ix = np.where((counts > lower_threshold) & (counts < upper_threshold), True, False)
    
    # Plot and format a weighted mito counts histogram
    sc.set_figure_params(dpi=100, figsize=[4,4], fontsize=12)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(counts, bins=100)
    ax.set_yscale('log')
    ax.set_xlabel('% Mitochondrial RNA counts per cell')
    ax.set_ylabel('# Cells per bin')
    ax.set_title(library_id)
    ax.text(0.99,0.95, str(np.sum(ix)) + '/' + str(counts.shape[0]) + ' cells retained', ha='right', va='center', transform=ax.transAxes)
    
    # Overlay the counts thresholds as vertical lines
    ax.plot([upper_threshold, upper_threshold], [0, ax.get_ylim()[1]])
    ax.plot([lower_threshold, lower_threshold], [0, ax.get_ylim()[1]])

    # Save figure to file
    fig.tight_layout()
    plt.savefig(save_path + 'mito_hist_' + library_id + '.png')
    plt.show()
    plt.close()

    # Print the number of cell barcodes that will be retained 
    print('Mito-Filtering ' + library_id + ' (' + str(np.sum(ix)) + '/' + str(counts.shape[0]) + ' cells retained)')
    print()

    # If requested, return a filtered version of adata
    if filter_cells:
        adata = adata[ix, :]
    
    return adata

    
def filter_ribo(adata, filter_cells=False, upper_threshold=100, lower_threshold=0, library_id='', save_path='./figures/'):
    '''
    Plots a weighted histogram of % ribosomal protein transcripts per cell barcode for guiding the
    placement of filtering thresholds. Returns a filtered version of adata if filter_cells=True.  
    '''

    # If necessary, create the output directory
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    # Use adata.uns['library_id'] if it exists
    if not library_id:
      if 'library_id' in adata.uns:
        library_id = adata.uns['library_id']

    # Calculate QC metric for % mitochondrial counts per cell
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS','rps','RPL','rpl'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['ribo'], inplace=True)
    counts = adata.obs['pct_counts_ribo']
    ix = np.where((counts > lower_threshold) & (counts < upper_threshold), True, False)
    
    #ix1 = counts < upper_threshold && counts > lower_threshold

    # Plot and format a weighted mito counts histogram
    sc.set_figure_params(dpi=100, figsize=[4,4], fontsize=12)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(counts, bins=100)
    ax.set_yscale('log')
    ax.set_xlabel('% Ribosomal Protein mRNA counts per cell')
    ax.set_ylabel('# Cells per bin')
    ax.set_title(library_id)
    ax.text(0.99,0.95, str(np.sum(ix)) + '/' + str(counts.shape[0]) + ' cells retained', ha='right', va='center', transform=ax.transAxes)
    
    # Overlay the counts thresholds as vertical lines
    ax.plot([upper_threshold, upper_threshold], [0, ax.get_ylim()[1]])
    ax.plot([lower_threshold, lower_threshold], [0, ax.get_ylim()[1]])

    # Save figure to file
    fig.tight_layout()
    plt.savefig(save_path + 'ribo_hist_' + library_id + '.png')
    plt.show()
    plt.close()

    # Print the number of cell barcodes that will be retained 
    print('Ribo-Filtering ' + library_id + ' (' + str(np.sum(ix)) + '/' + str(counts.shape[0]) + ' cells retained)')
    print()

    # If requested, return a filtered version of adata
    if filter_cells:
        adata = adata[ix, :]
    
    return adata


def filter_scrublet(adata, filter_cells=False, threshold=5):

    # disable copy data warning
    warnings.filterwarnings('ignore')

    # use adata.uns['library_id'] if it exists
    if 'library_id' in adata.uns:
      library_id = adata.uns['library_id']
    else:
      library_id = ''
  
    # calculate and plot doublet scores 
    sc.external.pp.scrublet(adata, threshold=threshold, verbose=False)
    sc.external.pl.scrublet_score_distribution(adata, scale_hist_sim='log')
    
    # print filtering summary
    print('Doublet Filtering ' + library_id + ' (' + str(len(adata) - sum(adata.obs['predicted_doublet'])) + '/' + str(adata.shape[0]) + ' cells retained)')
    print()
    
    if filter_cells:  
        adata = adata[~adata.obs['predicted_doublet'],:]

    return adata


