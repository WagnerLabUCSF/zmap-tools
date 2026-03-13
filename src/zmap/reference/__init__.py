from __future__ import annotations

"""
Reference data loading and consensus marker access.

Provides functions to download, cache, and load ZMAP reference H5AD datasets,
and to query curated consensus marker gene tables at multiple annotation levels.
"""

from .markers import load_consensus_markers
from .load_h5ad import load_zmap_h5ad

__all__ = ["load_consensus_markers", 
		   "load_zmap_h5ad"]
