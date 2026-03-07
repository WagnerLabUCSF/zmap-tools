from __future__ import annotations

from .dotplot_gene import gene_groups_vs_time, gene_groups_vs_studies, gene_groups_vs_time_and_studies 
from .dotplot_group import group_siblings_vs_markers, group_descendants_vs_markers

gene_view = gene_groups_vs_time_and_studies
group_view = group_siblings_vs_markers

__all__ = ["gene_groups_vs_time", 
		   "gene_groups_vs_studies",
		   "gene_groups_vs_time_and_studies",
		   "group_siblings_vs_markers",
		   "group_descendants_vs_markers",
		   "gene_view",
		   "group_view"]
