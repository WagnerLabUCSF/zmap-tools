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


__all__ = ["__version__", "ref", "dotplot", "predict"]
