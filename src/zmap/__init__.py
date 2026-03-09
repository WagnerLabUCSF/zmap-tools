from __future__ import annotations

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
