zmap-tools
==========

**zmap-tools** is the Python API for the Zebrafish Multi-Atlas Project (ZMAP) —
a curated single-cell RNA-seq reference atlas for zebrafish development.

It provides a simple **load → annotate → visualize** workflow:

.. code-block:: python

   import zmap

   # 1. Load reference
   adata_ref = zmap.ref.load_zmap_h5ad(kind="symphony")

   # 2. Annotate a query dataset
   zmap.predict.annotate_with_zmap(
       adata_query,
       query_raw_counts_source="counts",
       cluster_col="leiden",
   )

   # 3. Visualize marker genes
   zmap.dotplot.gene_view(adata_ref, "sox2")


Key Features
------------

- **Reference loading** — download and cache ZMAP H5AD files with persistent
  Google Drive integration for Colab workflows.
- **Label transfer** — kNN-based cell-type annotation using Symphony/Harmony
  embeddings, with per-cell confidence scoring and QC filters.
- **Consensus markers** — access curated marker gene tables ranked by
  specificity, contrast, consensus, and prevalence.
- **Dotplots** — publication-ready dotplots showing expression across
  cell types, timepoints, and studies.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/reference
   api/predict
   api/dotplot


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
