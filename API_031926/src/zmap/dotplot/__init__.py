from __future__ import annotations

"""
Dotplot visualization for ZMAP reference data.

Provides gene-centric dotplots (expression across cell types × timepoints
and studies) and group-centric dotplots (marker genes for a focal cell type
compared to its siblings and tissues).

Convenience aliases:

- ``gene_view`` → :func:`gene_groups_vs_time_and_studies`
- ``group_view`` → :func:`group_siblings_vs_markers`
"""

from .dotplot_gene import gene_groups_vs_time_and_studies 
from .dotplot_group import group_siblings_vs_markers, group_descendants_vs_markers

gene_view = gene_groups_vs_time_and_studies
group_view = group_siblings_vs_markers

__all__ = ["gene_groups_vs_time_and_studies",
		   "group_siblings_vs_markers",
		   "group_descendants_vs_markers",
		   "gene_view",
		   "group_view"]
