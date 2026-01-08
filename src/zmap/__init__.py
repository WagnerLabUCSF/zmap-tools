from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

# Package version (from the PyPI/dist name)
try:
    __version__ = _version("zmap-tools")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Import subpackages as attributes 
from . import process      # zmap.process
from . import reference    # zmap.reference
from . import tools        # zmap.tools
from . import dotplot      # zmap.dotplot
from . import predict      # zmap.predict

# Scanpy-style aliases: 
pp = process
ref = reference
tl = tools


__all__ = ["__version__", "pp", "ref", "tl" ,"dotplot", "predict"]
