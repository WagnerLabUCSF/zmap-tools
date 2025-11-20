from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

# Package version (from the PyPI/dist name)
try:
    __version__ = _version("zmap-tools")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Import subpackages so they become attributes on `zmap`
from . import preprocess   # zmap.preprocess
from . import ref          # zmap.ref
from . import dev          # zmap.dev

# Scanpy-style alias: zmap.pp -> zmap.preprocess
pp = preprocess

__all__ = ["__version__", "preprocess", "pp", "ref", "dev"]
