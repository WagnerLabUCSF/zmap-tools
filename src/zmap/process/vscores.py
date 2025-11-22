
import sys
import gc
import warnings
import numpy as np
import scipy
import sklearn
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# IDENTIFY HIGHLY VARIABLE GENES


def get_min_max_norm(data):

    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def runningquantile(x, y, p, nBins):
    """ calculate the quantile of y in bins of x """

    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    dx = (x[-1] - x[0]) / nBins
    xOut = np.linspace(x[0]+dx/2, x[-1]-dx/2, nBins)

    yOut = np.zeros(xOut.shape)

    for i in range(len(xOut)):
        ind = np.nonzero((x >= xOut[i]-dx/2) & (x < xOut[i]+dx/2))[0]
        if len(ind) > 0:
            yOut[i] = np.percentile(y[ind], p)
        else:
            if i > 0:
                yOut[i] = yOut[i-1]
            else:
                yOut[i] = np.nan

    return xOut, yOut


def get_vscores(E, min_mean=0, nBins=50, fit_percentile=0.1, error_wt=1):
    '''
    Calculate v-score (above-Poisson noise statistic) for genes in the input counts matrix
    Return v-scores and other stats
    '''

    ncell = E.shape[0]

    mu_gene = E.mean(axis=0).A.squeeze()
    gene_ix = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[gene_ix]

    tmp = E[:, gene_ix]
    tmp.data **= 2
    var_gene = tmp.mean(axis=0).A.squeeze() - mu_gene ** 2
    del tmp
    FF_gene = var_gene / mu_gene

    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene / mu_gene)

    x, y = runningquantile(data_x, data_y, fit_percentile, nBins)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    def gLog(input): return np.log(input[1] * np.exp(-input[0]) + input[2])
    h, b = np.histogram(np.log(FF_gene[mu_gene > 0]), bins=200)
    b = b[:-1] + np.diff(b) / 2
    max_ix = np.argmax(h)
    c = np.max((np.exp(b[max_ix]), 1))

    def errFun(b2): return np.sum(abs(gLog([x, c, b2]) - y) ** error_wt)
    b0 = 0.1
    b = scipy.optimize.fmin(func=errFun, x0=[b0], disp=False)
    a = c / (1 + b) - 1

    v_scores = FF_gene / ((1 + a) * (1 + b) + b * mu_gene)
    CV_eff = np.sqrt((1 + a) * (1 + b) - 1)
    CV_input = np.sqrt(b)

    return v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b


def get_vscores_adata(adata, norm_counts_per_cell=1e6, min_vscore_pctl=85, min_counts=3, min_cells=3, in_place=True):

    ''' 
    Identifies highly variable genes
    Requires a layer of adata that has been processed by total count normalization (we use tpm_nolog)
    '''

    E = adata.layers['tpm_nolog']
    
    # get variability statistics
    stats = {}
    stats['vscores'], stats['CV_eff'], stats['CV_input'], stats['gene_ix'], stats['mu_gene'], stats['FF_gene'], stats['a'], stats['b'] = get_vscores(E) # gene_ix = genes for which vscores could be returned
    stats['min_vscore_pctl'] = min_vscore_pctl

    # index genes based on vscores percentile
    ix2 = stats['vscores'] > 0 # ix2 = genes for which a positive vscore was obtained
    stats['min_vscore'] = np.percentile(stats['vscores'][ix2], min_vscore_pctl)    
    ix3 = (((E[:, stats['gene_ix'][ix2]] >= min_counts).sum(0).A.squeeze()>= min_cells) & (stats['vscores'][ix2] >= stats['min_vscore'])) # ix3 = genes passing final min cells & counts thresholds

    # highly variable genes = genes passing all 3 filtering steps: gene_ix, ix2, and ix3
    stats['hv_genes'] = adata.var_names[stats['gene_ix'][ix2][ix3]]
    
    if in_place:
      
        # save gene-level stats to adata.var
        adata.var['highly_variable'] = False
        adata.var.loc[stats['hv_genes'], 'highly_variable'] = True
        adata.var['vscore'] = np.nan
        adata.var.loc[adata.var_names[stats['gene_ix']], 'vscore'] = stats['vscores']
        adata.var['mu_gene'] = np.nan
        adata.var.loc[adata.var_names[stats['gene_ix']], 'mu_gene'] = stats['mu_gene']
        adata.var['ff_gene'] = np.nan
        adata.var.loc[adata.var_names[stats['gene_ix']], 'ff_gene'] = stats['FF_gene']
    
        # save vscore results to adata.uns
        adata.uns['vscore_stats'] = stats
        adata.uns['vscore_stats']['hv_genes'] = list(adata.uns['vscore_stats']['hv_genes'])

        return None
    
    else:
        
        # just return vscore stats
        return stats


def get_variable_genes(adata, batch_key=None, filter_method='all', top_n_genes=3000, norm_counts_per_cell=1e6, min_vscore_pctl=85, min_counts=3, min_cells=3, in_place=True):
    
    '''
    Filter variable genes based on their representation within individual sample batches
    '''

    # compute initial gene variability stats using the entire dataset
    get_vscores_adata(adata, norm_counts_per_cell=norm_counts_per_cell, min_vscore_pctl=min_vscore_pctl, min_counts=min_counts, min_cells=min_cells)
    
    # batch handling: if no batches are provided we are already done
    if batch_key == None: # 
        if in_place == True: 
            adata.var['vscore'] = get_min_max_norm(adata.var['vscore'])
            return adata
        else:
            return adata.var['highly_variable']
    
    # if multiple batches are present, then we will determine variable genes independently for each batch
    else:
        batch_ids = np.unique(adata.obs[batch_key])
        n_batches = len(batch_ids)

    # set hvgene filter method
    if filter_method == 'any':
        count_thresh = 0 # >0 = keep hvgenes identified in 1 or more batches
    elif filter_method == 'multiple':
        count_thresh = 1 # >1 = only keep hvgenes identified in 2 or more batches
    elif filter_method == 'majority':
        count_thresh = n_batches/2 # only keep hvgenes identified in >50% of batches
    elif filter_method == 'all':
        count_thresh = n_batches - 1 # only keep hvgenes identified in 100% of batches
    elif filter_method == 'top_n_genes': 
        min_vscore_pctl = 0 # return the top hv genes (# specified by 'top_n_genes') ranked by mean scaled vscore
    else:
        print('Invalid filtering method provided!')    

    # identify variable genes for each batch separately
    within_batch_hv_genes = []
    within_batch_vscores = np.full(shape=[adata.shape[1],n_batches], fill_value=np.nan)
    for n,b in enumerate(batch_ids):
        adata_batch = adata[adata.obs[batch_key] == b].copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            vscore_stats = get_vscores_adata(adata_batch, norm_counts_per_cell=norm_counts_per_cell, min_vscore_pctl=min_vscore_pctl, min_counts=min_counts, min_cells=min_cells, in_place=False)
        hv_genes_this_batch = list(vscore_stats['hv_genes'])
        within_batch_hv_genes.append(hv_genes_this_batch)
        within_batch_vscores[:,n] = get_min_max_norm(adata.var['vscore']) # scale vscores from 0 to 1

    # aggregate batch stats
    adata.var['vscore'] = np.nanmean(within_batch_vscores, axis=1) # return the mean of scaled vscores across all batches
    within_batch_hv_genes = [g for gene in within_batch_hv_genes for g in gene]
    within_batch_hv_genes, hv_batch_count = np.unique(within_batch_hv_genes, return_counts=True)
    
    # perform hv_gene filtering
    if filter_method == 'top_n_genes':
        hv_genes = adata.var['vscore'].sort_values(ascending=False)[0:top_n_genes].index
    else:
        hv_genes = within_batch_hv_genes[hv_batch_count > count_thresh]
        
    # update adata
    adata.var['highly_variable'] = False
    adata.var.loc[hv_genes, 'highly_variable'] = True    

    if in_place:
        return None
    else:
        return adata.var['highly_variable']


def plot_ff(adata, gene_ix=None, color=None):

  if gene_ix == None:
    gene_ix = adata.var['highly_variable']
  else:
    gene_ix = adata.var[gene_ix]

  if color == None:
    color = np.array(['blue'])

  mu_gene = adata.var['mu_gene']
  ff_gene = adata.var['ff_gene']
  a = adata.uns['vscore_stats']['a']
  b = adata.uns['vscore_stats']['b']

  x_min = 0.5 * np.min(mu_gene)
  x_max = 2 * np.max(mu_gene)
  xTh = x_min * np.exp(np.log(x_max / x_min) * np.linspace(0, 1, 100))
  yTh = (1 + a) * (1 + b) + b * xTh
  plt.figure(figsize=(6, 6))
  plt.scatter(np.log10(mu_gene), np.log10(ff_gene), c=np.array(['grey']), alpha=0.3, edgecolors=None, s=4)
  plt.scatter(np.log10(mu_gene)[gene_ix], np.log10(ff_gene)[gene_ix], c=color, alpha=0.3, edgecolors=None, s=4)

  plt.plot(np.log10(xTh), np.log10(yTh))
  plt.xlabel('Mean Transcripts Per Cell (log10)')
  plt.ylabel('Gene Fano Factor (log10)')
  plt.show()


def plot_vscores(adata, gene_ix=None, color=None):

  if gene_ix == None:
    gene_ix = adata.var['highly_variable']
  else:
    gene_ix = adata.var[gene_ix]
  
  if color == None:
    color = np.array(['blue'])
  
  mu_gene = adata.var['mu_gene']
  vscores_gene = adata.var['vscore']
  a = adata.uns['vscore_stats']['a']
  b = adata.uns['vscore_stats']['b']

  plt.figure(figsize=(6, 6))
  plt.scatter(np.log10(mu_gene), np.log10(vscores_gene), c=np.array(['grey']), alpha=0.3, edgecolors=None, s=4)
  plt.scatter(np.log10(mu_gene)[gene_ix], np.log10(vscores_gene)[gene_ix], c=color, alpha=0.3, edgecolors=None, s=4)

  plt.xlabel('Mean Transcripts Per Cell (log10)')
  plt.ylabel('Gene Vscores (log10)')
  plt.show()

 
def get_covar_genes(adata, minimum_correlation=0.2, show_hist=True):

    # Subset adata to highly variable genes x cells (counts matrix only)
    adata_tmp = sc.AnnData(adata[:,adata.var.highly_variable].X)

    # Determine if the input matrix is sparse
    sparse=False
    if scipy.sparse.issparse(adata_tmp.X):
      sparse=True

    # Get nn correlation distance for each highly variable gene
    gene_correlation_matrix = 1-sparse_corr(adata_tmp.X)
    np.fill_diagonal(gene_correlation_matrix, np.inf)
    max_neighbor_corr = 1-gene_correlation_matrix.min(axis=1)

    # filter genes whose nearest neighbor correlation is above threshold 
    ix_keep = np.array(max_neighbor_corr > minimum_correlation, dtype=bool).squeeze()
    
    # Prepare a randomized data matrix                    
    adata_tmp_rand = adata_tmp.copy()
    
    if sparse:
      mat = adata_tmp_rand.X.todense()
    else:
      mat = adata_tmp_rand.X
    
    # randomly permute each row of the counts matrix
    for c in range(mat.shape[1]):
        np.random.seed(seed=c)
        mat[:,c] = mat[np.random.permutation(mat.shape[0]),c]
    
    if sparse:        
      adata_tmp_rand.X = scipy.sparse.csr_matrix(mat)
    else:
      adata_tmp_rand.X = mat
    
    # Get nn correlation distances for randomized data
    gene_correlation_matrix = 1-sparse_corr(adata_tmp_rand.X)
    np.fill_diagonal(gene_correlation_matrix, np.inf)
    max_neighbor_corr_rand = 1-gene_correlation_matrix.min(axis=1)
    
    # Plot histogram of correlation distances
    plt.figure(figsize=(6, 6))
    plt.hist(max_neighbor_corr_rand, bins=np.linspace(0, 1, 100), density=False, alpha=0.5, label='random')
    plt.hist(max_neighbor_corr, bins=np.linspace(0, 1, 100), density=False, alpha=0.5, label='data')
    plt.axvline(x = minimum_correlation, color = 'k', linestyle = '--', alpha=0.5, linewidth=1)
    plt.xlabel('Nearest Neighbor Correlation')
    plt.ylabel('Counts')
    plt.legend(loc='upper right')
    plt.show()
    
    return adata_tmp.var[ix_keep]



