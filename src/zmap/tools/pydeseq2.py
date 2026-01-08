
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from adjustText import adjust_text

 

# DIFFERENTIAL EXPRESSION



def get_pydeseq2_sample_contrasts(adata, cluster_obs, sample_obs, condition_obs, condition_list, key_added='', csv_path=None):

    # Generate a pyDESeq2 results dataframe that reports cluster-level
    # pairwise comparisons (contrasts) between conditions over samples 
    #
    # cluster_obs:      column in adata.obs containing per cell cluster assignments
    # sample_obs:       column in adata.obs containing per cell sample assignments (e.g. 'Control_1', 'Mutant_1', etc)
    # condition_obs:    column in adata.obs containing per cell condition assignments (e.g. 'Control', 'Mutant')
    # condition_list:   list specifying condition order for comparisons (e.g. ['Mutant', 'Control']) 
    
    # Use a dictionary to store results
    pyDESeq_results = {}

    # Loop over all clusters
    clusters = list(np.unique(adata.obs[cluster_obs]))
    for cluster in clusters:
        print(cluster)
        adata_subset = adata[adata.obs[cluster_obs] == cluster]

        # Generate a set of pseudo-bulk profiles as adata objects - one for each sample (in each cluster)
        pb_adata_list = []
        for sample in np.unique(adata_subset.obs[sample_obs]):
            adata_subset_next = adata_subset[adata_subset.obs[sample_obs] == sample]
            del adata_subset_next.X
            adata_subset_next.X = adata_subset_next.layers['raw_nolog'] # make sure to use raw counts data
            pb_adata_next = sc.AnnData(X = np.array(adata_subset_next.X.sum(axis = 0)), var = adata_subset_next.var[[]])
            pb_adata_next.obs_names = [sample]
            pb_adata_next.obs[condition_obs] = adata_subset_next.obs[condition_obs].iloc[0]
            pb_adata_list.append(pb_adata_next)

        # Concatenate the sample-level pseudo-bulk adatas
        pb_adata = sc.concat(pb_adata_list)

        # Run pyDESeq2
        dds = DeseqDataSet(counts = pd.DataFrame(pb_adata.X, columns = pb_adata.var_names), 
                           metadata = pb_adata.obs, 
                           design_factors = 'condition',
                           quiet = True)
        dds.deseq2();
        stat_res = DeseqStats(dds, contrast=('condition', condition_list[0], condition_list[1]));
        stat_res.summary();

        # Sort the pyDESeq2 results table
        pyDESeq_results[cluster] = stat_res.results_df.sort_values('stat', ascending = False)

    # If requested, save results tables to csv
    if csv_path is not None:
        # Create the directory if it doesn't exist
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        print('Saving CSV tables')
        for cluster in clusters:
            pyDESeq_results[cluster].to_csv(csv_path + '/' + 'pyDESeq_results_' + str(cluster) + '.csv')

    # Store results in adata.uns
    adata.uns['pyDESeq2'+'-'+key_added] = pyDESeq_results
    return adata


def plot_pydeseq2_results_clustermap(adata, gene_list, cluster_obs, values_to_plot='log2FoldChange', metric='seuclidean', method='complete', key=None, cmap='vlag'):
    
    # Generate a dataframe to hold pydeseq2 results
    results_df = pd.DataFrame(index=gene_list, columns=adata.obs[cluster_obs].unique())
    for cluster in adata.obs[cluster_obs].unique():
        for g in gene_list:
            # Go into pydeseq2 results and extract values for each gene in each cluster
            if key is None:
                results_df.loc[g][cluster] = adata.uns['pyDESeq2'][cluster].loc[g][values_to_plot]
            else:
                results_df.loc[g][cluster] = adata.uns[key][cluster].loc[g][values_to_plot]
    results_df = results_df.astype(float).fillna(0)

    # Generate a Seaborn clustermap
    sns.set_style("white", {'axes.grid' : False})
    cg = sns.clustermap(results_df.T,
                      metric=metric, method=method,
                      cmap=cmap, vmin=-3, vmax=3,
                      figsize=(25,5), dendrogram_ratio=0.1, linewidths=0.5)
                      #cbar_kws={'label': 'Log2 Fold Change \n (RA vs Control)'})

    # Formatting
    cg.ax_heatmap.axhline(y=0, color='k', linewidth=1)
    cg.ax_heatmap.axhline(y=cg.data.shape[0], color='k', linewidth=1)
    cg.ax_heatmap.axvline(x=0, color='k',linewidth=1)
    cg.ax_heatmap.axvline(x=cg.data.shape[1], color='k', linewidth=1)
    cg.fig.subplots_adjust(right=0.7)
    cg.ax_cbar.set_position((0.8, .7, .01, .2))

    return cg


def plot_pydeseq2_cluster_sensitivities(adata, cluster_obs, sample_obs, condition_obs, condition_list, log2fc_threshold = 1, adj_pvalue_threshold = 0.05, key=None, return_dfs=False):
    
    # Compute normalized ratios of # cells in each cluster (total RA vs total control)
    #

    # Get crosstab of cell type clusters vs conditions
    ratios_df = pd.crosstab(adata.obs[cluster_obs], adata.obs[condition_obs])
    ratios_df

    # Normalize condition totals to cells per 10k cells
    condition_1 = condition_list[0]
    condition_2 = condition_list[1]
    nCells_1 = np.sum(adata.obs[condition_obs] == condition_1)
    nCells_2 = np.sum(adata.obs[condition_obs] == condition_2)
    ratios_df[condition_1] = ratios_df[condition_1]/nCells_1*10000
    ratios_df[condition_2] = ratios_df[condition_2]/nCells_2*10000

    # Get log2 ratio of normalized condition counts
    ratios_df['Ratio'] = np.log2(ratios_df[condition_1] / ratios_df[condition_2])
    ratios_df.sort_values(cluster_obs, ascending = True)

    # Import pyDESeq results
    if key is None:
        degs_df = adata.uns['pyDESeq2'].copy()
    else:
        degs_df = adata.uns[key].copy()

    # Get nDEGs and nCells for each cell type cluster
    power_df = pd.DataFrame(index=adata.obs[cluster_obs].unique(), columns=['nDEGs','nCells'])
    for cluster in adata.obs[cluster_obs].unique():
        degs_df[cluster] = degs_df[cluster].sort_values('log2FoldChange', ascending = False)
        flag_fc = np.logical_or(degs_df[cluster]['log2FoldChange']<-log2fc_threshold, degs_df[cluster]['log2FoldChange']>log2fc_threshold)
        flag_pv = degs_df[cluster]['padj']<adj_pvalue_threshold
        flag = np.logical_and(flag_fc, flag_pv)
        degs_df[cluster] = degs_df[cluster][flag]
        power_df['nDEGs'][cluster] = np.log1p(len(list(degs_df[cluster].index)))
        power_df['nCells'][cluster] = np.sum(adata.obs[cluster_obs]==cluster)

    
    # Compute the relative proportions of each cell type cluster
    power_df['cluster_size'] = list(power_df['nCells']/np.sum(power_df['nCells'])*100*10)

    # Generate scatterplot
    sns.set_style("white", {'axes.grid' : True})
    fig, ax = plt.subplots()
    sns.scatterplot(x = list(ratios_df['Ratio']), y = list(power_df['nDEGs']), s=power_df['cluster_size'])

     # Add and adjust point labels
    point_labels = [plt.annotate(label, (ratios_df['Ratio'][n], power_df['nDEGs'][n])) for n, label in enumerate(power_df.index)]
    adjust_text(point_labels, arrowprops=dict(arrowstyle="-", color='#1f77b4', lw=0.5))

    # Format axes
    plt.xlim(-1.5, 1.5)
    plt.xlabel('Log2 Cell Type Abundance (' + condition_1 + ' / ' + condition_2 + ')')
    plt.ylabel('Log10 nDEGs')

    if return_dfs:
        return ratios_df, power_df


