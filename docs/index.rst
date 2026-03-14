zmap-tools
==========

**zmap-tools** is the Python API for the Zebrafish Multi-Atlas Project (ZMAP) —
a curated single-cell RNA-seq reference atlas for zebrafish development.

It provides two core workflows:

**1. Explore the reference atlas** — load the ZMAP reference and browse
curated markers or generate publication-ready dotplots.

.. code-block:: python

   import zmap

   # Fetch data
   adata_ref = zmap.ref.load_zmap_h5ad()

   # Browse curated marker genes
   markers = zmap.ref.load_markers(level='CellType', groups=['diencephalon'], n_per_group=10, format='dict')

   # Visualize expression across cell types
   zmap.dotplot.gene_view(adata_ref, "sox2")

**2. Annotate a query dataset** — transfer ZMAP cell-type labels onto your
own single-cell data using kNN-based projection.

.. code-block:: python

   zmap.predict.annotate_with_zmap(
       adata_query,
       query_raw_counts_source="counts",
       cluster_col="leiden",
   )

Key Features
------------

- **Reference loading** — download and cache ZMAP H5AD files with persistent
  Google Drive integration for Colab workflows.
- **Consensus markers** — access curated marker gene tables ranked by
  specificity, contrast, consensus, and prevalence.
- **Dotplots** — publication-ready dotplots showing expression across
  cell types, timepoints, and studies.
- **Label transfer** — kNN-based cell-type annotation batch-corected
  embeddings, with per-cell confidence scoring and QC filters.

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