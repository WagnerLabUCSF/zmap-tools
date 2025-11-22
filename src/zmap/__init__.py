from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

# Package version (from the PyPI/dist name)
try:
    __version__ = _version("zmap-tools")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Import subpackages so they become attributes on `zmap`
from . import process      # zmap.process
from . import ref          # zmap.ref
from . import dev          # zmap.dev

# Scanpy-style alias: zmap.pp -> zmap.process
pp = process

__all__ = ["__version__", "process", "pp", "ref", "dev"]
