Quickstart
==========

This guide walks through the three core steps of a ZMAP workflow:
loading reference data, annotating a query dataset, and visualizing results.


Loading Reference Data
----------------------

The :func:`~zmap.reference.load_zmap_h5ad` function downloads and caches
ZMAP reference H5AD files. On Google Colab with Drive mounted, cache persists
across sessions.

.. code-block:: python

   import zmap

   # Default: processed_slim_tpm (best for visualization)
   adata_ref = zmap.ref.load_zmap_h5ad()

   # Symphony reference (required for label transfer)
   adata_sym = zmap.ref.load_zmap_h5ad(kind="symphony")

Available presets:

- ``"processed_slim_tpm"`` — TPM counts, best for plotting
- ``"processed_slim"`` — raw counts only
- ``"processed"`` — full dataset with intermediate layers
- ``"raw"`` — unprocessed raw counts
- ``"symphony"`` — Symphony reference for label transfer


Loading Consensus Markers
-------------------------

.. code-block:: python

   # Top 50 CellType markers as a dict
   markers = zmap.ref.load_consensus_markers()

   # Top 10 Tissue markers as a panel DataFrame (for dotplots)
   panel = zmap.ref.load_consensus_markers(
       level="Tissue",
       n_per_group=10,
       format="panel",
   )


Annotating a Query Dataset
--------------------------

The full pipeline — preprocess, embed, transfer labels, and plot — runs
in a single call:

.. code-block:: python

   adata_query = zmap.predict.annotate_with_zmap(
       adata_query,
       query_raw_counts_source="counts",   # where raw counts live
       cluster_col="leiden",               # your cluster column
   )

   # Results are in adata_query.obs:
   #   - ZMAP_CellType_predicted
   #   - ZMAP_CellType_predicted_prob
   #   - ZMAP_time_id

For finer control, the individual steps are also exposed:

.. code-block:: python

   # 1. Preprocess
   zmap.predict.preprocess_adata_query(adata_query, counts_source="counts")

   # 2. kNN label transfer (after Symphony embedding)
   zmap.predict.predict_labels_kNN(
       adata_query, adata_ref,
       ref_label_col="ZMAP_CellType",
       n_neighbors=25,
   )

   # 2b. Optional: tissue-aware kNN (same step-4 summary interface)
   zmap.predict.predict_label_tissue_kNN(
       adata_query, adata_ref,
       ref_label_col="ZMAP_CellType",
       tissue_mode="hard",
       knn_backend="auto",
   )

   # 3. Cluster-level summary
   df = zmap.predict.aggregate_by_cluster(
       adata_query,
       cluster_col="leiden",
       label_space="ZMAP_CellType",
   )


Dotplot Visualization
---------------------

Two-panel gene view (cell types × timepoints and studies):

.. code-block:: python

   zmap.dotplot.gene_view(adata_ref, "sox2")

Sibling comparison dotplot:

.. code-block:: python

   zmap.dotplot.group_view(adata_ref, "hepatocyte")

Descendant / sub-cluster dotplot:

.. code-block:: python

   zmap.dotplot.group_descendants_vs_markers(
       adata_ref,
       parent="forebrain",
       parent_col="ZMAP_Tissue",
       child_col="ZMAP_Cluster",
   )
