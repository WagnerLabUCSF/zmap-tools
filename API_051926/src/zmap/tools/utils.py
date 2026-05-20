
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import sklearn
import warnings
from scipy.sparse import coo_matrix
from scipy.stats import rankdata 



# UTILITIES


def get_smooth_values(adata, obs_use, k=15, n_rounds=1, rank=True, log=True, scale=True):

    # Format inputs
    graph = adata.obsp['connectivities'].tocoo()
    values = np.array(adata.obs[obs_use])
    n_nodes = graph.shape[0]
    
    # Convert values to ranks
    if rank:
        values = rankdata(values, method='dense')
        
    # Get slicing indices for each node along graph rows
    _, slice_idx = np.unique(graph.row, return_index=True)
    slice_idx = np.append(slice_idx, len(graph.col))
    

    # Perform specified # of rounds of smoothing
    for round in range(n_rounds):
        
        # Get smoothened values
        values_tmp = np.empty(n_nodes)
        for i in range(n_nodes):
            start_idx = slice_idx[i]
            end_idx = slice_idx[i+1]
            neighbor_indices = graph.col[start_idx:end_idx]
            neighbor_values = values[neighbor_indices]
            k_nearest_values = np.sort(neighbor_values)[:min(k, len(neighbor_values))]
            values_tmp[i] = np.nanmean(k_nearest_values)
        
        values = values_tmp
    
    if log: 
        values = np.log1p(values)

    if scale:
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
 

    return values
    

def plot_stacked_barplot(labels_A, labels_B, normalize='index', fig_width=4, fig_height=4):

    # Cross-tabulate the two sets of labels
    crstb = pd.crosstab(labels_A, labels_B, normalize=normalize)
    
    # Plot stacked bars
    crstb.plot.bar(stacked=True, width=0.8, figsize=(fig_width, fig_height))
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.ylim([0,1])
    plt.ylabel('Proportion')
    plt.grid(False)
    plt.show()


def get_deg_table(adata, ngenes_csv=100, ngenes_disp=20, thresh_pv=0.05, thresh_logfc=1, filt_hv=False, return_dict=False):
    
    # Uses results from rank_genes_groups
    
    # Convert the results dictionary to a dataframe with DEG names, log2 fold changes, adjusted p-values
    deg = adata.uns['rank_genes_groups']
    groups = list(deg['names'].dtype.names)
    df = pd.DataFrame({groups+'_'+key: deg[key][groups] for groups in groups for key in ['names','logfoldchanges','pvals_adj']}).head(ngenes_csv)
    df.to_csv('rank_genes_groups_DEGTable.csv')


    # Get list of highly variable genes, if requested
    if filt_hv:
      hv_genes = adata[1,adata.var['highly_variable']].var_names
    
    # Get list of markers for each group that pass filtering criteria
    markers=[]   # will be a list of lists 
    for g in groups:
        flag_log2fc = df[g+'_logfoldchanges'] > thresh_logfc
        flag_pv = df[g+'_pvals_adj'] < thresh_pv
        flag = flag_log2fc & flag_pv
        if filt_hv:
          flag_variable = df[g+'_names'].isin(hv_genes)
          flag = flag & flag_variable
        markers.append(list(df[g+'_names'][flag]))

    # Print to screen
    pd.options.display.max_columns = None
    dc = dict(zip(groups,markers))
    df = pd.DataFrame.from_dict(dc, orient='index').T
    df = df.head(ngenes_disp).fillna(value='')

    # return marker sets as a dictionary, if requested
    if return_dict:
      return dc
    else:
      return df
    
    
get_rank_genes_groups_table = get_deg_table
   