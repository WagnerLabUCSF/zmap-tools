
import scanpy as sc
import numpy as np
from .dimensionality import *


def adata2tpt(adata):

    # Perform TPT Normalization on X matrix of an adata object
    adata_tpt = adata.copy()
    adata_tpt.X = adata_tpt.layers['raw_nolog']
    sc.pp.normalize_total(adata_tpt, target_sum=1e4, inplace=True) 
    sc.pp.log1p(adata_tpt)
    
    # Confirm TPT
    #print(adata_tpt.X.expm1().sum(axis = 1))
    
    return adata_tpt
    

def pp_raw2norm(adata, include_raw_layers=True, include_tpm_layers=True):
	
	# Store raw counts as separate layer
	if include_raw_layers: adata.layers['raw_nolog'] = adata.X.copy()
	if include_raw_layers: adata.layers['raw'] = sc.pp.log1p(adata.X.copy())
	
	# Perform total counts normalization (tpm)
	sc.pp.normalize_total(adata, target_sum=1e6, inplace=True) # TPM Normalization

	# Perform log transformation and store tpm normalized counts in a separate layer
	if include_tpm_layers: adata.layers['tpm_nolog'] = adata.X.copy()
	sc.pp.log1p(adata)
	if include_tpm_layers: adata.layers['tpm'] = adata.X.copy()
	
	# Scale the genes by z-score (no zero center = large sparse matrix-friendly)
	sc.pp.scale(adata, zero_center=False)


def pp_norm2umap(adata, batch_key=None, n_neighbors=15, verbose=False, include_umap=True, include_leiden=True):

	# Perform PCA with a specified number of dimensions
	sc.pp.pca(adata, n_comps=adata.uns['n_sig_PCs'], zero_center=True)

	# Generate neighbor graph, incorporating Harmony integration if necessary
	if batch_key is not None:
		sc.external.pp.harmony_integrate(adata, batch_key, basis='X_pca', adjusted_basis='X_pca_harmony', max_iter_harmony=20, random_state=0, verbose=verbose)
		sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=adata.uns['n_sig_PCs'], metric='euclidean', use_rep='X_pca_harmony')
	else:
		sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=adata.uns['n_sig_PCs'], metric='euclidean', use_rep='X_pca')
	
	# Generate UMAP embedding
	if include_umap: sc.tl.umap(adata, n_components=2, spread=1)
	
	# Perform graph clustering
	if include_leiden: sc.tl.leiden(adata, resolution=1, key_added='leiden')


# LEGACY ALIASES
pp_get_embedding = pp_norm2umap