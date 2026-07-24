from __future__ import annotations

"""
zmap-tools — Python API for the Zebrafish Multi-Atlas Project (ZMAP).

A curated single-cell RNA-seq reference atlas for zebrafish development,
with tools for reference loading, label transfer, and visualization.

Submodules
----------
reference (aliased as ref)
    Download, cache, and load ZMAP reference H5ADs and consensus markers.
predict
    kNN-based label transfer from the ZMAP reference to query datasets.
dotplot
    Publication-ready dotplot visualization of gene expression patterns.
"""

from importlib.metadata import PackageNotFoundError, version as _version

# Package version (from the PyPI/dist name)
try:
    __version__ = _version("zmap-tools")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Import subpackages as attributes 
from . import reference    # zmap.reference
from . import dotplot      # zmap.dotplot
from . import predict      # zmap.predict

# Scanpy-style aliases: 
ref = reference

# Top-level convenience APIs for biological module / baseline2 workflows.
apply_encoder = predict.apply_encoder
zmap_projection = predict.zmap_projection
predict_pseudo_tissue = predict.predict_pseudo_tissue
knn_config = predict.knn_config
time_prediction_KNN = predict.time_prediction_KNN
time_prediction_MLP = predict.time_prediction_MLP
predict_labels_MLP = predict.predict_labels_MLP
predict_labels_mlp = predict.predict_labels_mlp
predict_labels_baseline2 = predict.predict_labels_baseline2
predict_labels_baseline_tissue = predict.predict_labels_baseline_tissue
baseline2 = predict.baseline2
baseline_tissue = predict.baseline_tissue
plot_conf_entropy_margin = predict.plot_conf_entropy_margin
plot_pseudo_tissue_conf_entropy_margin = predict.plot_pseudo_tissue_conf_entropy_margin
plot_confusion_matrix = predict.plot_confusion_matrix
report_leiden_celltype_topk = predict.report_leiden_celltype_topk
plot_leiden_celltype = predict.plot_leiden_celltype
plot_projection_celltype_overlay = predict.plot_projection_celltype_overlay

__all__ = ["__version__", "ref", "dotplot", "predict"]

__all__ += [
    "apply_encoder",
    "zmap_projection",
    "predict_pseudo_tissue",
    "knn_config",
    "time_prediction_KNN",
    "time_prediction_MLP",
    "predict_labels_MLP",
    "predict_labels_mlp",
    "predict_labels_baseline2",
    "predict_labels_baseline_tissue",
    "baseline2",
    "baseline_tissue",
    "plot_conf_entropy_margin",
    "plot_pseudo_tissue_conf_entropy_margin",
    "plot_confusion_matrix",
    "report_leiden_celltype_topk",
    "plot_leiden_celltype",
    "plot_projection_celltype_overlay",
]
