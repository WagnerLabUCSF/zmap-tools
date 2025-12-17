from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Literal, Sequence, Dict, List, Optional, Any

import pandas as pd
import requests

# ---------------------------------------------------------------------
# URL registry for your consensus tables
# ---------------------------------------------------------------------

_MARKER_URLS: Dict[str, str] = {
    "GermLayer":        "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/csv/consensus_markers/ZMAP_GermLayer_consensus_report.csv.zip",
    "Tissue":           "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/csv/consensus_markers/ZMAP_Tissue_consensus_report.csv.zip",
    "CellType":         "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/csv/consensus_markers/ZMAP_CellType_consensus_report.csv.zip",
    "CellTypeFine":     "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/csv/consensus_markers/ZMAP_CellTypeFine_consensus_report.csv.zip",
    "Cluster":          "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/csv/consensus_markers/ZMAP_Cluster_consensus_report.csv.zip",
    "Leiden100":        "https://pub-dbadc2c623224cb58d93cfa3b950fef5.r2.dev/csv/consensus_markers/ZMAP_leiden100_consensus_report.csv.zip",
}

# ---------------------------------------------------------------------
# Cache directory helper
# ---------------------------------------------------------------------

def _get_cache_dir() -> Path:
    """
    Return the local cache directory for zmap_tools marker tables.
    """
    root = Path(os.path.expanduser("~")) / ".cache" / "zmap_tools"
    root.mkdir(parents=True, exist_ok=True)
    return root

# ---------------------------------------------------------------------
# Download + load helper (handles macOS __MACOSX junk)
# ---------------------------------------------------------------------

def _load_marker_table(level: str) -> pd.DataFrame:
    """
    Internal: download + cache the consensus marker CSV.zip for a given level
    and return it as a pandas DataFrame.

    Handles extra __MACOSX/._ files in the ZIP (common on macOS).
    Also strips any leading "ZMAP_..._consensus_report" index-like column.
    """
    if level not in _MARKER_URLS:
        raise ValueError(f"Unknown level {level!r}. Must be one of {list(_MARKER_URLS.keys())}.")

    url = _MARKER_URLS[level]
    cache_dir = _get_cache_dir()
    zip_name = f"{level}_consensus_report.csv.zip"
    zip_path = cache_dir / zip_name

    # Download once, then reuse
    if not zip_path.exists():
        resp = requests.get(url)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    # Open zip and pick the real CSV (ignore __MACOSX junk)
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and not name.startswith("._")
        ]

        if len(csv_files) == 0:
            raise ValueError(f"No CSV file found inside {zip_name}. Files: {z.namelist()}")

        if len(csv_files) > 1:
            raise ValueError(
                f"Multiple CSV files found in {zip_name}, ambiguous: {csv_files}. "
                "Expected exactly one real CSV in the ZIP."
            )

        csv_name = csv_files[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    # Clean column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    # Some of your files have a leading column like "ZMAP_Tissue_consensus_report"
    # which is just an index-like artifact. Drop it if present.
    first_col = df.columns[0]
    if first_col.startswith("ZMAP_") or first_col.startswith("Unnamed:"):
        df = df.drop(columns=[first_col])

    return df


# ---------------------------------------------------------------------
# Unannotated gene filtering
# ---------------------------------------------------------------------

_UNANNOTATED_PATTERNS = [
    r"^ENSDARG\d+",     # Ensembl gene IDs
    r"^si:",            # common clone / placeholder names
    r"^sb:",            # common clone / placeholder names
    r"^im:",            # common clone / placeholder names
    r"^zgc:",           # ZGC placeholders
    r"^zmp:",           # ZMP placeholders
    r"^LOC\d+",         # predicted loci
    r"^XLOC",           # predicted loci
    r"^linc",           # long noncoding placeholders
    r"^wu:",            # WU clone names
    r"^bx",             # BX clone IDs
    r"^GRCz\d+_",       # assembly scaffolds like GRCz11_...
    r"^CR\d+",          # CR293511.1, CR936442.1
    r"^AL\d+",          # AL845362.1
    r"^CABZ\d+",        # CABZ01074130.1
    r"^CT\d+",          # CT737162.3
    r"^FO\d+",          # FO704741.1
    r"^CU\d+",          # 
    r"^FP\d+",          # 
    r"^FQ\d+",          # 
    r"^LO\d+",          #    
]

_UNANNOTATED_REGEX = re.compile("|".join(_UNANNOTATED_PATTERNS), re.IGNORECASE)


def _filter_unannotated_genes(df: pd.DataFrame, gene_col: str = "gene") -> pd.DataFrame:
    """
    Remove genes that appear to be unannotated or placeholder names.

    This includes:
      - missing or empty gene names
      - Ensembl IDs (ENSDARG...)
      - common zebrafish placeholder prefixes:
        si:, zgc:, LOC, linc, wu:, bx, GRCz...
    """
    if gene_col not in df.columns:
        return df

    # Drop NaN / empty
    df = df.dropna(subset=[gene_col])
    df = df[df[gene_col].astype(str).str.strip() != ""]

    # Remove placeholder patterns
    mask = df[gene_col].astype(str).apply(
        lambda g: not bool(_UNANNOTATED_REGEX.match(g))
    )
    return df[mask]


# ---------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------

def _df_to_dict(df: pd.DataFrame, group_col: str, gene_col: str) -> Dict[str, List[str]]:
    """
    Convert long-form table to {group: [gene1, gene2, ...]}.
    """
    out: Dict[str, List[str]] = {}
    for g, sub in df.groupby(group_col, sort=False):
        out[str(g)] = sub[gene_col].astype(str).tolist()
    return out


def _df_to_sets(df: pd.DataFrame, group_col: str, gene_col: str) -> Dict[str, set]:
    """
    Convert long-form table to {group: set([...])}.
    """
    out: Dict[str, set] = {}
    for g, sub in df.groupby(group_col, sort=False):
        out[str(g)] = set(sub[gene_col].astype(str).tolist())
    return out


def _df_to_panel(df: pd.DataFrame, group_col: str, gene_col: str) -> pd.DataFrame:
    """
    Minimal dotplot design format for downstream plotting:
        group | gene
    """
    return (
        df[[group_col, gene_col]]
        .copy()
        .rename(columns={group_col: "group"})
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------

def load_consensus_markers(
    level: Literal["Tissue", "CellType", "Cluster", "Leiden100"] = "CellType",
    *,
    groups: Optional[Sequence[str]] = None,
    marker_type: Literal["specificity", "contrast", "consensus", "overall"] = "overall",
    n_per_group: Optional[int] = 50,
    min_support_ratio: Optional[float] = None,
    min_log2fc: Optional[float] = None,
    min_enrich: Optional[float] = None,
    omit_unannotated: bool = False,
    format: Literal["dict", "sets", "table", "panel"] = "dict",
) -> Any:
    """
    Load ZMAP consensus markers for a chosen reference level.

    Parameters
    ----------
    level
        Which annotation level to load:
        {"Tissue", "CellType", "Cluster", "Leiden100"}.
    groups
        Optional subset of groups (e.g. only certain tissues/cell types).
        Uses the `celltype` column in the table, which is the "group" at that level.
    marker_type
        Which ranking column to use when selecting top markers:

        - "specificity" -> use 'rank_specificity'
        - "contrast"    -> use 'rank_contrast'
        - "consensus"   -> use 'rank_consensus'
        - "overall"     -> use 'overall_rank'

    n_per_group
        Number of markers to return per group. If None, return all markers
        passing the filters for each group.
    min_support_ratio
        If provided, keep only rows with support_ratio >= this value.
    min_log2fc
        If provided, keep only rows with global_log2fc >= this value.
    min_enrich
        If provided, keep only rows with enrich_mean >= this value.
    omit_unannotated
        If True, remove genes that look like unannotated/placeholder IDs
        (Ensembl IDs, si:, zgc:, LOC..., linc..., wu:, bx..., GRCz...).
    format
        Output format:

        - "dict"  -> {group: [marker1, marker2, ...]}
        - "sets"  -> {group: set([...])}
        - "table" -> filtered long-form DataFrame
        - "panel" -> small DataFrame with columns ["group", "gene"]

    Returns
    -------
    object
        Depends on `format`:

        - dict:  Dict[str, List[str]]
        - sets:  Dict[str, Set[str]]
        - table: pd.DataFrame
        - panel: pd.DataFrame
    """
    df = _load_marker_table(level)

    group_col = "celltype"
    gene_col = "gene"

    if group_col not in df.columns or gene_col not in df.columns:
        raise KeyError(
            f"Expected columns {group_col!r} and {gene_col!r} in consensus table for level={level!r}.\n"
            f"Found columns: {list(df.columns)}"
        )

    # Optional: remove unannotated/placeholder genes
    if omit_unannotated:
        df = _filter_unannotated_genes(df, gene_col=gene_col)

    # Restrict to a subset of groups
    if groups is not None:
        groups = [str(g) for g in groups]
        df = df[df[group_col].astype(str).isin(groups)]

    # Basic filters
    if (min_support_ratio is not None) and ("support_ratio" in df.columns):
        df = df[df["support_ratio"] >= float(min_support_ratio)]

    if (min_log2fc is not None) and ("global_log2fc" in df.columns):
        df = df[df["global_log2fc"] >= float(min_log2fc)]

    if (min_enrich is not None) and ("enrich_mean" in df.columns):
        df = df[df["enrich_mean"] >= float(min_enrich)]

    # Choose ranking column
    marker_to_col = {
        "specificity": "rank_specificity",
        "contrast": "rank_contrast",
        "consensus": "rank_consensus",
        "overall": "overall_rank",
    }
    rank_col = marker_to_col[marker_type]
    if rank_col not in df.columns:
        raise KeyError(
            f"Requested marker_type={marker_type!r}, but column {rank_col!r} "
            f"is not present in the consensus table. Available columns: {list(df.columns)}"
        )

    # Select top-N per group by the chosen rank
    if n_per_group is not None:
        df = (
            df.sort_values([group_col, rank_col], kind="mergesort")
              .groupby(group_col, group_keys=False)
              .head(int(n_per_group))
        )

    # Return in the requested format
    if format == "table":
        return df.reset_index(drop=True)

    if format == "dict":
        return _df_to_dict(df, group_col, gene_col)

    if format == "sets":
        return _df_to_sets(df, group_col, gene_col)

    if format == "panel":
        return _df_to_panel(df, group_col, gene_col)

    raise ValueError(f"Unknown format {format!r}. Must be one of 'dict', 'sets', 'table', 'panel'.")
