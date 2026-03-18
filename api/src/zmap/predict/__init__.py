from __future__ import annotations

"""
Label transfer, annotation, and diagnostic visualization.

Provides the end-to-end ``annotate_with_zmap`` pipeline as well as
lower-level functions for preprocessing, kNN-based label transfer,
cluster aggregation, and UMAP overlay plotting.
"""

from .predict import (
    preprocess_adata_query,
    predict_label_tissue_kNN,
    predict_labels_kNN,
    summarize_knn_run,
    aggregate_by_cluster,
    build_cell_annotations_table,
    plot_colorbar_histogram,
    sync_zmap_colors,
    plot_embedding_with_ondata_labels,
    map_query_labels,
    annotate_with_zmap,
)

__all__ = [
    "preprocess_adata_query",
    "predict_label_tissue_kNN",
    "predict_labels_kNN",
    "summarize_knn_run",
    "aggregate_by_cluster",
    "build_cell_annotations_table",
    "plot_colorbar_histogram",
    "sync_zmap_colors",
    "plot_embedding_with_ondata_labels",
    "map_query_labels",
    "annotate_with_zmap",
]
