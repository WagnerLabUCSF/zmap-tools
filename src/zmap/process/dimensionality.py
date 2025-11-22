
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



# IDENTIFY SIGNIFICANT PCA DIMENSIONS


def get_sig_pcs(adata, n_iter = 3, nPCs_test = 300, threshold_method='95', show_plots=True, zero_center=True, verbose=True, in_place=True):

    # Subset adata to highly variable genes x cells (counts matrix only)
    adata_tmp = sc.AnnData(adata[:,adata.var.highly_variable].X)

    # Determine if the input matrix is sparse
    sparse=False
    if scipy.sparse.issparse(adata_tmp.X):
      sparse=True

    # Get eigenvalues from pca on data matrix
    if verbose: 
        print('Performing PCA on data')
    sc.pp.pca(adata_tmp, n_comps=nPCs_test, zero_center=zero_center)
    eig = adata_tmp.uns['pca']['variance']

    # Get eigenvalues from pca on randomly permuted data matrices
    if verbose: 
        print('Performing PCA on randomized data')
    eig_rand = np.zeros(shape=(n_iter, nPCs_test))
    eig_rand_max = []
    n_sig_PCs_trials = []
    for j in range(n_iter):

        if verbose and n_iter>1: sys.stdout.write('\rIteration %i / %i' % (j+1, n_iter)); sys.stdout.flush()
        
        adata_tmp_rand = adata_tmp.copy()
        
        if sparse:
          mat = adata_tmp_rand.X.todense()
        else:
          mat = adata_tmp_rand.X
        
        # randomly permute each row of the counts matrix
        for c in range(mat.shape[1]):
            np.random.seed(seed=j+c)
            mat[:,c] = mat[np.random.permutation(mat.shape[0]),c]
        
        if sparse:        
          adata_tmp_rand.X = scipy.sparse.csr_matrix(mat)
        else:
          adata_tmp_rand.X = mat
        
        sc.pp.pca(adata_tmp_rand, n_comps=nPCs_test, zero_center=zero_center)
        eig_rand_next = adata_tmp_rand.uns['pca']['variance']
        eig_rand[j,:] = eig_rand_next
        eig_rand_max.append(np.max(eig_rand_next))
        n_sig_PCs_trials.append(np.count_nonzero(eig>np.max(eig_rand_next)))

        del adata_tmp_rand
        gc.collect()


    # Set eigenvalue thresholding method
    if threshold_method == '95':
        method_string = 'Counting the # of PCs with eigenvalues above random in >95% of trials'
        eig_thresh = np.percentile(eig_rand_max,95)
    elif threshold_method == 'median':
        method_string = 'Counting the # of PCs with eigenvalues above random in >50% of trials'
        eig_thresh = np.percentile(eig_rand_max,50)
    elif threshold_method == 'all':
        method_string = 'Counting the # of PCs with eigenvalues above random across all trials'
        eig_thresh = np.percentile(eig_rand_max,100)
    
    # Determine # of PC dimensions with eigenvalues above threshold
    n_sig_PCs = np.count_nonzero(eig>eig_thresh)    

    if show_plots: 

        # Plot eigenvalue histograms
        bins = np.logspace(0, np.log10(np.max(eig)+10), 50)
        sns.histplot(eig_rand.flatten(), bins=bins, kde=False, alpha=1, label='random', stat='probability', color='orange')#, weights=np.zeros_like(data_rand) + 1. / len(data_rand))
        sns.histplot(eig, bins=bins, kde=False, alpha=0.5, label='data', stat='probability')#, weights=np.zeros_like(data) + 1. / len(data))
        plt.legend(loc='upper right')
        plt.axvline(x = eig_thresh, color = 'k', linestyle = '--', alpha=0.5, linewidth=1)
        plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.show()

        # Plot scree (eigenvalues for each PC dimension)
        plt.plot([], label='data', color='#1f77b4', alpha=1)
        plt.plot([], label='random', color='#ff7f0e', alpha=1)
        plt.plot(eig, alpha=1, color='#1f77b4')
        for j in range(n_iter):
          plt.plot(eig_rand[j], alpha=1/n_iter, color='#ff7f0e')   
        plt.legend(loc='upper right')
        plt.axhline(y = eig_thresh, color = 'k', linestyle = '--', alpha=0.5, linewidth=1)
        plt.yscale('log')
        plt.xlabel('PC #')
        plt.ylabel('Eigenvalue')
        plt.show()

        # Plot nPCs above rand histograms
        sns.set_context(rc = {'patch.linewidth': 0.0})
        sns.histplot(n_sig_PCs_trials, kde=True, stat='probability', color='#1f77b4') 
        plt.xlabel('# PCs Above Random')
        plt.ylabel('Frequency')
        plt.xlim([0, nPCs_test])
        plt.show()

    # Print summary stats to screen
    if verbose: 
        print()
        print(method_string)
        print('Eigenvalue Threshold =', np.round(eig_thresh, 2))
        print('# Significant PCs =', n_sig_PCs)

    if in_place:
        adata.uns['n_sig_PCs'] = n_sig_PCs
        adata.uns['n_sig_PCs_trials'] = n_sig_PCs_trials
        gc.collect()
        return None
    
    else:
        gc.collect()
        return n_sig_PCs, n_sig_PCs_trials



# LEGACY ALIASES
get_significant_pcs = get_sig_pcs

