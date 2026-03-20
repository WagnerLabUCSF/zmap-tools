``zmap.dotplot`` — Dotplot Visualization
=========================================

.. module:: zmap.dotplot

Publication-ready dotplots for exploring gene expression across cell types,
developmental timepoints, and studies.

The module provides two entry points via convenient aliases:

- ``zmap.dotplot.gene_view`` → :func:`gene_groups_vs_time_and_studies`
- ``zmap.dotplot.group_view`` → :func:`group_siblings_vs_markers`


Gene-Centric Dotplots
---------------------

.. autofunction:: gene_groups_vs_time_and_studies

.. autofunction:: zmap.dotplot.dotplot_gene.gene_groups_vs_time

.. autofunction:: zmap.dotplot.dotplot_gene.gene_groups_vs_studies


Group-Centric Dotplots
----------------------

.. autofunction:: group_siblings_vs_markers

.. autofunction:: group_descendants_vs_markers


Low-Level Engine
----------------

.. autofunction:: zmap.dotplot.dotplot_group.plot_dotplot_basegrid
