
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
    

def get_confusion_matrix(labels_A, labels_B, normalize=True, title=None, reorder_columns=True, reorder_rows=True, cmap=plt.cm.Blues, overlay_values=False, vmin=None, vmax=None, show_plot=True, return_df=False, figsize=4):

    # Filter labels if value is missing from either set
    nan_flag = labels_A.isnull() | labels_B.isnull()
    labels_A = labels_A[~nan_flag]
    labels_B = labels_B[~nan_flag]

    # Get all the unique values for each set of labels
    labels_A_unique = np.unique(labels_A.astype('string'))
    labels_B_unique = np.unique(labels_B.astype('string'))

    # Compute confusion matrix 
    cm = sklearn.metrics.confusion_matrix(labels_A, labels_B)
    non_empty_rows = cm.sum(axis=0)!=0
    non_empty_cols = cm.sum(axis=1)!=0
    cm = cm[:,non_empty_rows]
    cm = cm[non_empty_cols,:]
    cm = cm.T
    
    # Normalize by rows (label B)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Set title, colorbar, and axis names
    if normalize:
        colorbar_label = 'Fraction Overlap'
        if not title:
            title = 'Normalized confusion matrix'
    else:
        colorbar_label = '# Overlaps'
        if not title:
            title = 'Confusion matrix, without normalization'  
  
    # If available, get the label category names for plotting
    if hasattr(labels_A, 'name'):
        labels_A_name = labels_A.name 
    else:
        labels_A_name = 'Label A'
    if hasattr(labels_B, 'name'):
        labels_B_name = labels_B.name 
    else:
        labels_B_name = 'Label B'

    # If requested, reorder the rows and columns by best match
    if reorder_columns:
        top_match_c = np.argmax(cm, axis=0)
        reordered_columns = np.argsort(top_match_c)
        cm=cm[:,reordered_columns]
        labels_A_unique_sorted = labels_A_unique[reordered_columns]
    else:
        labels_A_unique_sorted = labels_A_unique

    if reorder_rows:
        top_match_r = np.argmax(cm, axis=1)
        reordered_rows = np.argsort(top_match_r)
        cm=cm[reordered_rows,:]
        labels_B_unique_sorted = labels_B_unique[reordered_rows]
    else:
        labels_B_unique_sorted = labels_B_unique

    # If requested, generate heatmap and format figure axes
    if show_plot:
        
        plt.rcParams['axes.grid'] = False
        fig, ax = plt.subplots(figsize=(figsize,figsize))
        im = plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal') 
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_title(title)
        ax.set_ylabel(labels_B_name)
        ax.set_xlabel(labels_A_name)
        ax.set_xticklabels(labels_A_unique_sorted, rotation=90, ha='center', minor=False)
        ax.set_yticklabels(labels_B_unique_sorted)

        # Format colorbar
        cb=ax.figure.colorbar(im, ax=ax, shrink=0.5)
        #cb.ax.tick_params(labelsize=10) 
        cb.ax.set_ylabel(colorbar_label, rotation=90)

        # If requested, loop over data dimensions and create text annotations
        if overlay_values:
            fmt = '.1f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            size=8)
    
    # If requested, return dataframe mapping top A match for each B
    if return_df:
    
        if reorder_rows:
            labels_A_mapped = labels_A_unique_sorted[top_match_r]
        else:
            labels_A_mapped = labels_A_unique_sorted

        mapping_df = pd.DataFrame(data=labels_A_mapped, index=labels_B_unique, columns=['top_match'])
        
        # Sort the index labels, if possible
        orig_list = labels_B_unique
        orig_list_digits = [s for s in orig_list if s.isdigit()]
        if len(orig_list)==len(orig_list_digits):
            mapping_df.index = mapping_df.index.astype(int)
            mapping_df = mapping_df.sort_index()
        
        return mapping_df
        
            
plot_confusion_matrix = get_confusion_matrix # alias to legacy function name 


def propagate_labels(adata, obs_from, obs_to, new_obs_name=None):

    with warnings.catch_warnings():
      warnings.simplefilter('ignore')

      # Group the dataframe by the 'a' column and find the mode of the 'b' column within each group
      most_common_per_group = adata.obs.groupby(obs_to)[obs_from].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()

      # Rename the columns
      if new_obs_name==None:
        new_obs_name = obs_from + '->' + obs_to
      most_common_per_group.columns = [obs_to, new_obs_name]

      # If target column name already exists, we will replace it
      if new_obs_name in adata.obs:
        del adata.obs[new_obs_name]

      # Merge w/the original DataFrame
      result_df = pd.merge(adata.obs, most_common_per_group, on=obs_to, how='left')
      result_df[new_obs_name] = result_df[new_obs_name].astype('category')

      # Save to adata
      adata.obs = result_df

    return adata


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
   