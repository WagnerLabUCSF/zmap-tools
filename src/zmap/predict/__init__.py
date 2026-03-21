from __future__ import annotations

"""
Label transfer, annotation, and diagnostic visualization.

Provides the end-to-end ``annotate_with_zmap`` pipeline as well as
lower-level functions for preprocessing, kNN-based label transfer,
cluster aggregation, and UMAP overlay plotting.

On-demand plot accessors (``plot_qc``, ``plot_embedding``, ``plot_time``,
``plot_overlap_matrix``, ``show_summary``) let you re-display results from a
completed run without recomputing.
"""

from .predict import (
    preprocess_adata_query,
    predict_labels_tissue_kNN,
    predict_labels_kNN,
    summarize_knn_run,
    aggregate_by_cluster,
    build_cell_annotations_table,
    plot_colorbar_histogram,
    sync_zmap_colors,
    plot_embedding_with_ondata_labels,
    map_query_labels,
    annotate_with_zmap,
    # On-demand accessors
    plot_qc,
    plot_embedding,
    plot_time,
    plot_overlap_matrix,
    show_summary,
    validate_markers,
)

__all__ = [
    "preprocess_adata_query",
    "predict_labels_tissue_kNN",
    "predict_labels_kNN",
    "summarize_knn_run",
    "aggregate_by_cluster",
    "build_cell_annotations_table",
    "plot_colorbar_histogram",
    "sync_zmap_colors",
    "plot_embedding_with_ondata_labels",
    "map_query_labels",
    "annotate_with_zmap",
    # On-demand accessors
    "plot_qc",
    "plot_embedding",
    "plot_time",
    "plot_overlap_matrix",
    "show_summary",
    "validate_markers",
]
