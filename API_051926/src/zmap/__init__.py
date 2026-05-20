from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version
import warnings

# Package version (from the PyPI/dist name)
try:
    __version__ = _version("zmap-tools")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Import core subpackages as attributes.
from . import process      # zmap.process
from . import reference    # zmap.reference
from . import dotplot      # zmap.dotplot
from . import predict      # zmap.predict

# Optional subpackage: may require extra dependencies (e.g., pydeseq2).
try:
    from . import tools  # zmap.tools
except ModuleNotFoundError as exc:
    tools = None  # type: ignore[assignment]
    warnings.warn(
        "Optional submodule 'zmap.tools' could not be imported. "
        f"Missing dependency: {exc}. Install extra deps to use zmap.tl.",
        RuntimeWarning,
        stacklevel=2,
    )

# Scanpy-style aliases:
pp = process
ref = reference
tl = tools

# Top-level convenience APIs for encoder projection + tissue-aware kNN.
apply_encoder = predict.apply_encoder
zmap_projection = predict.zmap_projection
predict_pseudo_tissue = predict.predict_pseudo_tissue
predict_labels_kNN = predict.predict_labels_kNN
predict_labels_mlp = predict.predict_labels_mlp
knn_config = predict.knn_config
plot_conf_entropy_margin = predict.plot_conf_entropy_margin
plot_pseudo_tissue_conf_entropy_margin = predict.plot_pseudo_tissue_conf_entropy_margin
plot_confusion_matrix = predict.plot_confusion_matrix
report_leiden_celltype_topk = predict.report_leiden_celltype_topk
plot_leiden_celltype = predict.plot_leiden_celltype
plot_projection_celltype_overlay = predict.plot_projection_celltype_overlay
time_prediction = predict.time_prediction


__all__ = [
    "__version__",
    "pp",
    "ref",
    "tl",
    "dotplot",
    "predict",
    "apply_encoder",
    "zmap_projection",
    "predict_pseudo_tissue",
    "predict_labels_kNN",
    "predict_labels_mlp",
    "knn_config",
    "plot_conf_entropy_margin",
    "plot_pseudo_tissue_conf_entropy_margin",
    "plot_confusion_matrix",
    "report_leiden_celltype_topk",
    "plot_leiden_celltype",
    "plot_projection_celltype_overlay",
    "time_prediction",
]
