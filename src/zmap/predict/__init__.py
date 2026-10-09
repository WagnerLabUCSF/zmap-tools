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
    plot_marker_comparison,
)
from .encoder_pipeline import (
    apply_encoder,
    zmap_projection,
    predict_pseudo_tissue,
    knn_config,
    build_module_faiss,
    build_module_concat_reference,
    apply_module_concat_query,
    time_prediction_KNN,
    plot_conf_entropy_margin,
    plot_pseudo_tissue_conf_entropy_margin,
    plot_confusion_matrix,
    report_leiden_celltype_topk,
    plot_leiden_celltype,
    plot_projection_celltype_overlay,
)
from .baseline2 import (
    predict_labels_MLP,
    predict_labels_mlp,
    predict_labels_baseline2,
    predict_labels_baseline_tissue,
    time_prediction_MLP,
    baseline2,
    baseline_tissue,
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
    "plot_marker_comparison",
]

__all__ += [
    "apply_encoder",
    "zmap_projection",
    "predict_pseudo_tissue",
    "knn_config",
    "build_module_faiss",
    "build_module_concat_reference",
    "apply_module_concat_query",
    "time_prediction_KNN",
    "plot_conf_entropy_margin",
    "plot_pseudo_tissue_conf_entropy_margin",
    "plot_confusion_matrix",
    "report_leiden_celltype_topk",
    "plot_leiden_celltype",
    "plot_projection_celltype_overlay",
    "predict_labels_MLP",
    "predict_labels_mlp",
    "predict_labels_baseline2",
    "predict_labels_baseline_tissue",
    "time_prediction_MLP",
    "baseline2",
    "baseline_tissue",
]
