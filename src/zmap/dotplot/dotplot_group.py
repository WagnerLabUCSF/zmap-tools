from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence, Mapping, Tuple

import re
import ast

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
import anndata as ad

from mpl_toolkits.axes_grid1 import make_axes_locatable  # kept if used elsewhere
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from zmap.reference import load_consensus_markers

# ---------------------------------------------------------------------
# Level configuration (ontology levels and their parents)
# ---------------------------------------------------------------------
_ZMAP_LEVEL_CONFIG = {
    "ZMAP_GermLayer": {
        "parent_col": "ZMAP_GermLayer",
        "consensus_level": "GermLayer",
    },
    "ZMAP_Tissue": {
        "parent_col": "ZMAP_GermLayer",
        "consensus_level": "Tissue",
    },
    "ZMAP_CellType": {
        "parent_col": "ZMAP_Tissue",
        "consensus_level": "CellType",
    },
    "ZMAP_CellTypeFine": {
        "parent_col": "ZMAP_Tissue",
        "consensus_level": "CellTypeFine",
    },
    "ZMAP_Cluster": {
        "parent_col": "ZMAP_CellTypeFine",
        "consensus_level": "Cluster",
    },
}

_ZMAP_LEVEL_LOOKUP = {
    "ZMAP_Tissue":       "Tissue",
    "ZMAP_CellType":     "CellType",
    "ZMAP_CellTypeFine": "CellTypeFine",
    "ZMAP_Cluster":      "Cluster",
}


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def _add_top_sibling_axes(ax, *, height_in: float, pad_in: float = 0.0):
    """
    Create a NEW axes directly above `ax` with fixed height in inches.
    Width matches `ax`. Returns the new axes.
    """
    fig = ax.figure
    fig_w, fig_h = fig.get_size_inches()
    base = ax.get_position()  # in figure fraction
    h_frac = float(height_in) / fig_h
    pad_frac = float(pad_in) / fig_h
    left, right = base.x0, base.x1
    bottom = base.y1 + pad_frac
    height = h_frac
    return fig.add_axes([left, bottom, right - left, height])


def _pt(val: float) -> float:
    """Convert points to inches for inset_axes / fig.add_axes math."""
    return float(val) / 72.0


def _inset_fixed(ax, *, width, height, loc, bbox_to_anchor=(0, 0, 1, 1), borderpad=0.0):
    """
    Create an inset axes with fixed physical size anchored to 'ax'.
    width/height can be floats (inches) or strings like '100%'.
    """
    return inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        borderpad=borderpad,
    )


def _add_left_time_strip_inset(
    fig,
    ax_main,
    *,
    adata,
    groupby: str,
    row_order: list[str],
    time_key: str = "time_id",
    vmin: float = 0.0,
    vmax: float = 120.0,
    cmap: str = "grey",
    width_pt: float = 12.0,           # strip width (points)
    gap_pt: float = 4.0,              # gap between y-axis and strip (points)
    show_yticklabels: bool = False,
    label_colors: dict[str, str] | None = None,
    draw_border: bool = False,
    ylabel: str = "Time (hpf)",
    ylabel_fontsize: float = 7.0,
    ylabel_color: str = "0.4",
):
    """
    Add a vertical time strip (median time per group) to the LEFT of ax_main.

    Both the strip width and the gap between the main y-axis and the strip are
    specified in physical units (points), so the geometry is stable across
    different figure sizes.
    """

    if groupby not in adata.obs.columns:
        raise KeyError(f"groupby '{groupby}' not found in adata.obs.")
    if time_key not in adata.obs.columns:
        raise KeyError(f"time_key '{time_key}' not found in adata.obs.")

    df = adata.obs[[groupby, time_key]].dropna(subset=[groupby, time_key])
    if df.empty:
        return None

    med = df.groupby(groupby)[time_key].median()

    missing = [g for g in row_order if g not in med.index]
    if missing:
        raise ValueError(
            f"The following groups in row_order have no valid '{time_key}' values: {missing}"
        )

    med = med.reindex(row_order)
    arr = med.to_numpy(dtype=float)[:, None]

    # --- Compute physical sizes (inches) and convert to axes-fraction offset ---
    strip_width_in = _pt(width_pt)   # strip width in inches
    gap_in = _pt(gap_pt)             # gap between y-axis and strip in inches

    fig_w, fig_h = fig.get_size_inches()
    pos = ax_main.get_position()     # [x0, y0, width, height] in figure fraction
    ax_width_in = fig_w * pos.width if fig_w > 0 else 0.0

    if ax_width_in > 0:
        # axes x=0 is at the main y-axis; negative fraction shifts left
        left_edge_frac = - (gap_in + strip_width_in) / ax_width_in
    else:
        left_edge_frac = 0.0

    # Fixed-size inset: strip width in inches, height spans the dotplot axis box
    ax_strip = _inset_fixed(
        ax_main,
        width=strip_width_in,       # inches
        height="100%",
        loc="lower left",
        bbox_to_anchor=(left_edge_frac, 0.0, 1, 1),
        borderpad=0.0,
    )

    im = ax_strip.imshow(
        arr,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    nrows = len(row_order)
    ax_strip.set_ylim(-0.5, nrows - 0.5)

    ax_strip.grid(False)
    ax_strip.set_axisbelow(False)
    ax_strip.set_xticks([])
    if show_yticklabels:
        ax_strip.set_yticks(np.arange(len(row_order)))
        ax_strip.set_yticklabels(row_order, fontsize=8)
        if label_colors is not None:
            for lab in ax_strip.get_yticklabels():
                t = lab.get_text()
                if t in label_colors:
                    lab.set_color(label_colors[t])
    else:
        ax_strip.set_yticks([])
        ax_strip.set_yticklabels([])

    if draw_border:
        for s in ax_strip.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.0)
            s.set_color("black")
    else:
        for s in ax_strip.spines.values():
            s.set_visible(False)

    if ylabel:
        ax_strip.text(
            0.5,
            -0.005,
            ylabel,
            ha="center",
            va="top",
            rotation=90,
            fontsize=ylabel_fontsize,
            color=ylabel_color,
            transform=ax_strip.transAxes,
        )

    return ax_strip, im


def find_level_for_node(
    adata: ad.AnnData,
    node: str,
    candidate_cols: Optional[Sequence[str]] = None,
) -> str:
    """
    Return the name of the obs column where `node` appears as a value.

    If found in multiple columns, choose the *highest* level according to the
    ordering in _ZMAP_LEVEL_CONFIG.
    """
    if candidate_cols is None:
        candidate_cols = list(_ZMAP_LEVEL_CONFIG.keys())  # highest → lowest

    hits: list[str] = []
    node_str = str(node)

    for col in candidate_cols:
        if col in adata.obs.columns:
            vals = adata.obs[col].astype(str)
            if node_str in set(vals):
                hits.append(col)

    if not hits:
        raise ValueError(
            f"{node_str!r} not found in any candidate level columns: {candidate_cols!r}."
        )

    if len(hits) > 1:
        for col in candidate_cols:
            if col in hits:
                return col

    return hits[0]


def get_parent_and_siblings(
    adata: ad.AnnData,
    node: str,
    level_col: str,
    parent_col: Optional[str],
) -> tuple[Optional[str], list[str]]:
    """
    Return (parent_label, siblings) for a given node in `level_col`.

    - If level_col == "ZMAP_Tissue":
        siblings are *all* tissues in ZMAP_Tissue.
        parent_label is inferred from parent_col if possible, else None.

    - Else:
        siblings are all values in `level_col` that share the same parent
        in `parent_col`. The returned `siblings` list includes the focal node.
    """
    if level_col not in adata.obs.columns:
        raise KeyError(f"level_col={level_col!r} not found in adata.obs.")

    node_str = str(node)

    # Special case: tissue-level → siblings = all tissues
    if level_col == "ZMAP_Tissue":
        vals = (
            adata.obs[level_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if node_str not in vals:
            raise ValueError(
                f"{node_str!r} not found in level_col={level_col!r} when treating as top-level."
            )

        siblings = sorted(vals)

        parent_label: Optional[str] = None
        if parent_col is not None and parent_col in adata.obs.columns:
            df = adata.obs[[level_col, parent_col]].dropna()
            df[level_col] = df[level_col].astype(str)
            df[parent_col] = df[parent_col].astype(str)
            parents = df.loc[df[level_col] == node_str, parent_col].unique()
            if len(parents) == 1:
                parent_label = parents[0]

        return parent_label, siblings

    # Normal case: use parent_col
    if parent_col is None:
        raise ValueError(
            f"parent_col is None for level_col={level_col!r}; cannot infer siblings."
        )
    if parent_col not in adata.obs.columns:
        raise KeyError(f"parent_col={parent_col!r} not found in adata.obs.")

    df = adata.obs[[level_col, parent_col]].dropna()
    df[level_col] = df[level_col].astype(str)
    df[parent_col] = df[parent_col].astype(str)

    parents = df.loc[df[level_col] == node_str, parent_col].unique()
    if len(parents) == 0:
        raise ValueError(
            f"No rows with {level_col} == {node_str!r} when looking for parent in {parent_col!r}."
        )
    if len(parents) > 1:
        raise ValueError(
            f"{node_str!r} has multiple parents in {parent_col!r}: {parents!r}."
        )

    parent_label = parents[0]
    siblings = (
        df.loc[df[parent_col] == parent_label, level_col]
        .astype(str)
        .unique()
        .tolist()
    )
    siblings = sorted(siblings)

    return parent_label, siblings


def get_node_markers(
    node: str,
    level_col: str,
    *,
    marker_types: Iterable[Literal["overall", "exclusivity", "contrast", "consensus"]] = ("overall",),
    n_per_type: int = 10,
    min_support_ratio: float | None = None,
    min_log2fc: float | None = None,
    min_enrich: float | None = None,
    omit_unannotated: bool = False,
) -> pd.DataFrame:
    """
    Load consensus markers for a single node across one or more marker_types.
    Returns a concatenated DataFrame with at least columns: ['celltype', 'gene', 'marker_type']
    """
    if level_col not in _ZMAP_LEVEL_CONFIG:
        raise KeyError(
            f"level_col={level_col!r} not recognized in _ZMAP_LEVEL_CONFIG."
        )

    consensus_level = _ZMAP_LEVEL_CONFIG[level_col]["consensus_level"]

    all_frames: list[pd.DataFrame] = []
    for mtype in marker_types:
        df = load_consensus_markers(
            level=consensus_level,
            groups=[str(node)],
            marker_type=mtype,
            n_per_group=n_per_type,
            min_support_ratio=min_support_ratio,
            min_log2fc=min_log2fc,
            min_enrich=min_enrich,
            omit_unannotated=omit_unannotated,
            format="table",
        )
        if df is None or df.empty:
            continue
        df = df.copy()
        df["marker_type"] = mtype
        df["celltype"] = df["celltype"].astype(str)
        df["gene"] = df["gene"].astype(str)
        all_frames.append(df)

    if not all_frames:
        mt = list(marker_types)
        raise ValueError(
            f"No markers returned for node={node!r}, level={consensus_level!r}, "
            f"marker_types={mt}."
        )

    markers = pd.concat(all_frames, ignore_index=True)
    return markers


def make_sibling_design_df(
    node: str,
    siblings: Sequence[str],
    marker_df: pd.DataFrame,
    *,
    genes: Sequence[str] | None = None,
    support_col: str = "support_ratio",
) -> pd.DataFrame:
    """
    Construct a design_df with columns ['celltype', 'gene', support_col]
    for use with plot_dotplot_design_fullgrid, where:

      - rows are all (sibling, gene) pairs
      - if `genes` is provided, it is treated as the canonical column order
        and ONLY genes in this list (that are present in marker_df for `node`)
        are included.
      - if `genes` is None, genes come from node_markers['gene'].unique()
        in that order.
      - support_col is populated for the focal node (if present in marker_df),
        and NaN for other siblings.
    """
    marker_df = marker_df.copy()
    marker_df["gene"] = marker_df["gene"].astype(str)
    marker_df["celltype"] = marker_df["celltype"].astype(str)

    node_str = str(node)
    node_markers = marker_df.loc[marker_df["celltype"] == node_str]
    if node_markers.empty:
        raise ValueError(f"No rows in marker_df for focal node={node_str!r}.")

    if genes is None:
        # Fallback: use the intrinsic order of genes for this node
        genes_list = node_markers["gene"].unique().tolist()
    else:
        # Enforce provided canonical order; keep only genes present for this node
        genes_set = set(node_markers["gene"].unique().tolist())
        genes_list = [str(g) for g in genes if str(g) in genes_set]
        if not genes_list:
            raise ValueError(
                f"None of the requested genes are present for node={node_str!r} "
                f"in marker_df."
            )

    # Cross product: siblings × genes (in canonical genes_list order)
    design_df = pd.DataFrame(
        [(ct, g) for ct in siblings for g in genes_list],
        columns=["celltype", "gene"],
    )

    # Propagate support values for the focal node only
    if support_col in node_markers.columns:
        sup_map = (
            node_markers
            .drop_duplicates(["celltype", "gene"])
            .set_index("gene")[support_col]
        )
        design_df[support_col] = np.where(
            design_df["celltype"] == node_str,
            design_df["gene"].map(sup_map),
            np.nan,
        )
    else:
        design_df[support_col] = np.nan

    return design_df


def reorder_groups_by_mean_expression(
    adata: ad.AnnData,
    groups: Sequence[str],
    *,
    level_col: str,
    genes: Sequence[str],
    layer: str | None = None,
    use_raw: bool | None = None,
) -> list[str]:
    """
    Reorder groups so that rows are sorted by the *average expression*
    across all provided genes, in descending order.

    Mean is taken over all cells in that group × all genes in `genes`.
    """
    groups = [str(s) for s in groups]
    if len(groups) <= 1 or len(genes) == 0:
        return groups

    if level_col not in adata.obs.columns:
        raise KeyError(f"level_col={level_col!r} not found in adata.obs.")

    # Choose expression matrix
    if use_raw and getattr(adata, "raw", None) is not None:
        X = adata.raw.X
        var_names = pd.Index(adata.raw.var_names).astype(str)
    elif layer is not None:
        X = adata.layers[layer]
        var_names = pd.Index(adata.var_names).astype(str)
    else:
        X = adata.X
        var_names = pd.Index(adata.var_names).astype(str)

    genes = pd.Index(genes).astype(str)
    gene_idx = var_names.get_indexer(genes)
    valid = gene_idx >= 0
    genes = genes[valid]
    gene_idx = gene_idx[valid]

    if genes.empty:
        return groups

    Xg = X[:, gene_idx]
    Xg = Xg.toarray() if hasattr(Xg, "toarray") else np.asarray(Xg)

    labels = adata.obs[level_col].astype(str).values

    row_means: dict[str, float] = {}
    for s in groups:
        idx = np.where(labels == s)[0]
        if idx.size == 0:
            continue
        sub = Xg[idx, :]
        m = float(np.nanmean(sub))
        if np.isfinite(m):
            row_means[s] = m

    if not row_means:
        return groups

    groups_sorted = sorted(
        groups,
        key=lambda s: row_means.get(s, -np.inf),
        reverse=True,
    )
    return groups_sorted


def reorder_genes_by_mean_expression(
    adata: ad.AnnData,
    genes: Sequence[str],
    *,
    groups: Sequence[str],
    level_col: str,
    layer: str | None = None,
    use_raw: bool | None = None,
) -> list[str]:
    """
    Reorder genes so that columns are sorted by the average expression
    across all cells in the specified `groups` (descending).

    Mean is taken over all cells with obs[level_col] in `groups`,
    across each gene in `genes`.
    """
    genes = [str(g) for g in genes]
    if len(genes) <= 1:
        return genes

    if level_col not in adata.obs.columns:
        raise KeyError(f"level_col={level_col!r} not found in adata.obs.")

    # choose expression matrix
    if use_raw and getattr(adata, "raw", None) is not None:
        X = adata.raw.X
        var_names = pd.Index(adata.raw.var_names).astype(str)
    elif layer is not None:
        X = adata.layers[layer]
        var_names = pd.Index(adata.var_names).astype(str)
    else:
        X = adata.X
        var_names = pd.Index(adata.var_names).astype(str)

    genes_idx = var_names.get_indexer(genes)
    valid = genes_idx >= 0
    genes = [g for g, v in zip(genes, valid) if v]
    genes_idx = genes_idx[valid]

    if not genes:
        return []

    Xg = X[:, genes_idx]
    Xg = Xg.toarray() if hasattr(Xg, "toarray") else np.asarray(Xg)

    labels = adata.obs[level_col].astype(str).values

    # collect all indices for the requested groups
    idx_list = [np.where(labels == g)[0] for g in groups]
    idx = np.concatenate([i for i in idx_list if i.size > 0]) if idx_list else np.array([], dtype=int)
    if idx.size == 0:
        return genes

    means = np.nanmean(Xg[idx, :], axis=0)
    order = np.argsort(-means)     # descending
    genes_sorted = [genes[i] for i in order]
    return genes_sorted

# ---------------------------------------------------------------------
# Primary DotPlot Helper Function
# ---------------------------------------------------------------------

def plot_dotplot_design_fullgrid(
    adata,
    design_df: pd.DataFrame,                 # must include 'celltype' and 'gene'; optional 'support'
    *,
    groupby: str = "ZMAP_CellType",
    layer: str | None = None,
    use_raw: bool | None = None,
    detect_threshold: float = 0.0,           # > threshold => “expressing”
    # ring settings
    support_col: str = "support_ratio",
    add_support_ring: bool = True,
    ring_min_lw: float = 0.02,
    ring_max_lw: float = 1.5,
    ring_color: str = "darkorange",
    ring_alpha: float = 0.8,
    # color/size encodings
    cmap: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    standard_scale: str | None = None,       # None | "var" | "obs"
    size_mode: str = "sqrt",               # "sqrt" or "linear"
    s_min: float = 2,
    s_max: float = 40,
    # row label suffix
    rowlabel_append_child_counts: bool = True,
    rowlabel_child_col: str = "ZMAP_Cluster",
    rowlabel_fmt: str = "{name} ({n})",
    # figure
    figsize=(12, 12),
    xlabel_rotation: int = 90,
    title: str | None = None,
    add_colorbar: bool = True,
    cbar_title: str = "log(tpm)\nCounts",
    cbar_title_fontsize: float = 8.0,
    cbar_ticklabel_fontsize: float = 8.0,

    # ===================== DOT SIZE LEGEND =====================
    show_size_legend: bool = True,
    size_legend_fracs: tuple = (0.1, 0.25, 0.5, 0.75, 1.0),
    size_legend_title: str = "Fraction\nexpressing",
    size_legend_loc: str = "upper left",
    size_legend_bbox_to_anchor: tuple | None = (0.0, 0.7),
    size_legend_label_fmt: str = "{:.0%}",
    size_legend_facecolor: str = "black",
    size_legend_edgecolor: str = "black",
    size_legend_edge_lw: float = 0.1,
    size_legend_framealpha: float = 0.9,
    size_legend_title_fontsize: float = 8.0,
    size_legend_label_fontsize: float = 8.0,

    # ===================== SUPPORT RING LEGEND =====================
    show_ring_legend: bool = True,
    ring_legend_fracs: tuple = (0, 0.5, 1.0),
    ring_legend_title: str = "Consensus\nsupport",
    ring_legend_loc: str = "lower left",
    ring_legend_bbox_to_anchor: tuple | None = (0.0, 0.25),
    ring_legend_label_fmt: str = "{:.0%}",
    ring_legend_title_fontsize: float = 8.0,
    ring_legend_label_fontsize: float = 8.0,

    # ===================== CONSENSUS PANEL (pick one) =====================
    consensus_panel: str | None = None,      # None | "line" | "support_grid" | "stacked_bar" | "strip"
    support_stack_field: str = "supporting_study_names",

    # ---- Line panel options ("line") ----
    support_line_agg: str = "median",        # "median" | "mean"
    support_line_color: str = "black",
    support_line_lw: float = 1,
    support_line_marker: str = '',
    support_line_ms: float = 0,
    support_line_alpha: float = 0.9,
    support_line_ylim: tuple | None = (0.0, 1.2),

    # ---- Support GRID options ("support_grid") ----
    support_grid_show_legend: bool = False,
    support_grid_legend_ncols: int = 2,

    # ---- Stacked BAR options ("stacked_bar") ----
    stacked_bar_show_legend: bool = True,
    stacked_bar_legend_ncols: int = 2,

    # ---- Strip options ("strip") ----
    support_strip_agg: str = "median",
    support_strip_cmap: str = "grey",
    support_strip_vmin: float | None = 0.0,
    support_strip_vmax: float | None = 1.0,
    support_strip_show_colorbar: bool = False,

    # ===== Palette sampling from a continuous cmap for study colors =====
    support_palette_from_cmap: str | mpl.colors.Colormap | None = None,
    support_palette_range: tuple[float, float] = (0.1, 0.9),

    # ===== Group color & duplicate gene columns =====
    group_color_dict: dict[str, str] | None = None,
    duplicate_gene_columns: bool = True,

    # ===== Vertical group separators =====
    draw_group_separators: bool = True,
    separator_color: str = "0.88",
    separator_lw: float = 0.7,
    separator_ls: str = "-",
    include_outer_separators: bool = False,

    # ===== Left time strip (inset) ==================================
    left_time_strip: bool = False,
    time_key: str = "time_id",
    time_cmap: str = "jet",
    time_vmin: float = 0.0,
    time_vmax: float = 120.0,
    yticklabel_gap_pt: float = 18.0,   # gap (points) between ticklabels and dotplot
    time_strip_show_yticklabels: bool = False,
    time_strip_label_colors: dict[str, str] | None = None,

    # ===================== PHYSICAL SIZE CONTROLS =====================
    # consensus/top panel fixed height (in inches)
    consensus_height_in: float = 0.35,
    # support grid/bar specific heights (in inches); if None, use consensus_height_in
    support_grid_height_in: float | None = None,
    stacked_bar_height_in: float | None = None,
    strip_height_pt: float = 12.0,  # for 'strip' variant
    # left time strip fixed width (points)
    time_strip_width_pt: float = 12.0,
    time_strip_gap_pt: float = 4.0,
    # colorbar physical sizing
    cbar_width_pt: float = 12.0,
    cbar_height_frac: float = 0.90,  # relative to dotplot height
    cbar_bbox_to_anchor: tuple | None = (1.05, 0.85),
    # reserve right gutter for legends/colorbar
    right_gutter_frac: float = 0.82,

    # ===== NEW: external axes control =====
    ax: plt.Axes | None = None,
    ax_leg: plt.Axes | None = None,
    adjust_right_gutter: bool = True,
):
    # --------------------- validation & ordering ---------------------
    if not {"celltype", "gene"}.issubset(design_df.columns):
        raise ValueError("design_df must contain columns: 'celltype' and 'gene'.")
    if standard_scale not in (None, "var", "obs"):
        raise ValueError("standard_scale must be None, 'var', or 'obs'.")
    if consensus_panel not in (None, "line", "support_grid", "stacked_bar", "strip"):
        raise ValueError(
            "consensus_panel must be one of: None, 'line', 'support_grid', 'stacked_bar', 'strip'."
        )
    if not (0.0 <= support_palette_range[0] < support_palette_range[1] <= 1.0):
        raise ValueError("support_palette_range must be within [0,1] and low < high.")

    if groupby not in adata.obs.columns:
        raise KeyError(f"groupby '{groupby}' not found in adata.obs.")

    if left_time_strip and time_key not in adata.obs.columns:
        raise KeyError(f"time_key '{time_key}' not found in adata.obs.")

    if consensus_panel in ("support_grid", "stacked_bar"):
        if support_stack_field not in design_df.columns:
            raise ValueError(
                f"consensus_panel='{consensus_panel}' requires column "
                f"'{support_stack_field}' in design_df."
            )

    design = design_df.copy()
    design["celltype"] = design["celltype"].astype(str)
    design["gene"] = design["gene"].astype(str)

    def _ordered_unique(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # row order
    row_order = _ordered_unique(design["celltype"].tolist())
    row_index = {r: i for i, r in enumerate(row_order)}

    # column slots
    if duplicate_gene_columns:
        cols_df = (
            design.drop_duplicates(subset=["celltype", "gene"], keep="first")
            .loc[:, ["celltype", "gene"]]
            .copy()
        )
        cols_df.rename(columns={"celltype": "col_owner"}, inplace=True)
        cols_df["col_id"] = np.arange(len(cols_df))
        col_labels = cols_df["gene"].tolist()
    else:
        col_order = _ordered_unique(design["gene"].tolist())
        cols_df = pd.DataFrame({"col_owner": [None] * len(col_order), "gene": col_order})
        cols_df["col_id"] = np.arange(len(col_order))
        col_labels = col_order

    col_count = len(cols_df)
    col_index = {i: i for i in range(col_count)}

    # --------------------- choose expression matrix ---------------------
    if use_raw and getattr(adata, "raw", None) is not None:
        X = adata.raw.X
        var_names = pd.Index(adata.raw.var_names).astype(str)
    elif layer is not None:
        X = adata.layers[layer]
        var_names = pd.Index(adata.var_names).astype(str)
    else:
        X = adata.X
        var_names = pd.Index(adata.var_names).astype(str)

    # --------------------- build grid ---------------------
    grid = (
        pd.MultiIndex.from_product(
            [row_order, cols_df["col_id"].tolist()],
            names=["celltype", "col_id"],
        )
        .to_frame(index=False)
        .merge(cols_df[["col_id", "gene", "col_owner"]], on="col_id", how="left")
    )

    if support_col in design.columns:
        sup_map = design.groupby(["celltype", "gene"], as_index=False)[support_col].first()
        grid = grid.merge(sup_map, on=["celltype", "gene"], how="left")
    else:
        grid[support_col] = np.nan

    grid["mean_expr"] = np.nan
    grid["frac_pos"] = np.nan

    # --------------------- compute stats ---------------------
    present_genes = pd.Index(cols_df["gene"].unique()).intersection(var_names)
    jmap = pd.Series(range(len(var_names)), index=var_names)
    labels = adata.obs[groupby].astype(str).values
    ct_to_idx = {ct: np.where(labels == ct)[0] for ct in row_order}

    batch = 256
    present_list = present_genes.tolist()
    for start in range(0, len(present_list), batch):
        sub_genes = present_list[start : start + batch]
        cols = jmap.loc[sub_genes].to_numpy()
        X_dense = (
            X[:, cols].toarray()
            if hasattr(X, "tocsc") or hasattr(X, "tocsr")
            else np.asarray(X[:, cols])
        )

        for ct, idx in ct_to_idx.items():
            if idx.size == 0:
                continue
            sub = X_dense[idx, :]
            means = np.asarray(sub.mean(axis=0)).ravel()
            fracs = np.asarray((sub > detect_threshold).mean(axis=0)).ravel()
            m = pd.Series(means, index=sub_genes)
            f = pd.Series(fracs, index=sub_genes)

            sel = (grid["celltype"] == ct) & (grid["gene"].isin(sub_genes))
            if not sel.any():
                continue
            grid.loc[sel, "mean_expr"] = grid.loc[sel, "gene"].map(m)
            grid.loc[sel, "frac_pos"] = grid.loc[sel, "gene"].map(f)

    grid["_ri"] = grid["celltype"].map(row_index)
    grid["_cj"] = grid["col_id"].map(col_index)

    # --------------------- optional standard-scale ---------------------
    if standard_scale is not None:
        mat_df = grid.pivot(
            index="celltype",
            columns="col_id",
            values="mean_expr",
        ).reindex(index=row_order, columns=cols_df["col_id"].tolist())
        mat = mat_df.to_numpy(dtype=float)

        def _scale_0_1(a, axis):
            amin = np.nanmin(a, axis=axis, keepdims=True)
            amax = np.nanmax(a, axis=axis, keepdims=True)
            rng = amax - amin
            with np.errstate(invalid="ignore", divide="ignore"):
                scaled = (a - amin) / rng
            scaled = np.where(np.isfinite(scaled), scaled, 0.0)
            scaled = np.where(rng == 0, 0.0, scaled)
            return scaled

        mat_scaled = _scale_0_1(mat, axis=0 if standard_scale == "var" else 1)
        scaled_df = pd.DataFrame(
            mat_scaled, index=row_order, columns=cols_df["col_id"].tolist()
        )
        scaled_long = (
            scaled_df.stack()
            .rename("mean_expr_scaled")
            .reset_index()
            .rename(columns={"level_0": "celltype", "level_1": "col_id"})
        )
        grid = grid.merge(scaled_long, on=["celltype", "col_id"], how="left")
        grid["mean_expr"] = grid["mean_expr_scaled"]
        grid.drop(columns=["mean_expr_scaled"], inplace=True)

    # --------------------- color/size scaling ---------------------
    vals = grid["mean_expr"].astype(float).to_numpy()
    if vmin is None:
        vmin = np.nanpercentile(vals, 1) if np.isfinite(vals).any() else 0.0
    if vmax is None:
        vmax = np.nanpercentile(vals, 99) if np.isfinite(vals).any() else 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    sz_raw = grid["frac_pos"].astype(float).clip(lower=0.0)
    sz_scale = np.sqrt(sz_raw) if size_mode == "sqrt" else sz_raw
    sz = (
        s_min
        + (s_max - s_min) * (sz_scale / np.nanmax(sz_scale))
        if np.nanmax(sz_scale) > 0
        else np.full_like(sz_scale, s_min)
    )

    # --------------------- main dot grid ---------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        if figsize is not None:
            fig.set_size_inches(figsize, forward=True)

    # Group separators behind dots
    col_count = len(cols_df)
    if draw_group_separators and col_count > 1:
        owners = cols_df["col_owner"].tolist()
        boundaries = [j for j in range(1, col_count) if owners[j] != owners[j - 1]]
        if include_outer_separators:
            boundaries = [0] + boundaries + [col_count]
        for j in boundaries:
            ax.axvline(
                x=j - 0.5,
                color=separator_color,
                linewidth=separator_lw,
                linestyle=separator_ls,
                zorder=0,
            )

    # Dots
    ax.scatter(
        grid["_cj"],
        grid["_ri"],
        s=sz,
        c=cmap_obj(norm(grid["mean_expr"].astype(float))),
        edgecolor="black",
        linewidth=0.1,
    )

    # -----------------------------------------------------------------
    # Reserve / use right gutter (for legends/colorbar) and create ax_leg
    # -----------------------------------------------------------------
    fig = ax.get_figure()

    if ax_leg is None:
        if adjust_right_gutter:
            fig.subplots_adjust(right=right_gutter_frac)

        base = ax.get_position()  # after any subplots_adjust
        gutter_left = base.x1 + 0.01
        gutter_right = 1.0 - 0.02
        gutter_width = max(0.01, gutter_right - gutter_left)
        ax_leg = fig.add_axes([gutter_left, base.y0, gutter_width, base.height])
        ax_leg.axis("off")
    # else: use provided ax_leg as-is

    # Support ring overlay
    if add_support_ring and support_col in grid.columns:
        sr = grid[support_col].astype(float).clip(0.0, 1.0).fillna(0.0)
        lw = ring_min_lw + (ring_max_lw - ring_min_lw) * sr
        mask = lw.to_numpy() > 0
        if np.any(mask):
            ax.scatter(
                grid.loc[mask, "_cj"],
                grid.loc[mask, "_ri"],
                s=sz[mask],
                facecolors="none",
                edgecolors=ring_color,
                linewidths=lw[mask],
                alpha=ring_alpha,
            )

    # Cosmetics
    xpad = 0.3
    ax.set_xlim(-0.5 - xpad, col_count - 0.5 + xpad)
    ax.set_ylim(len(row_order) - 0.5, -0.5)

    ax.set_xticks(range(col_count))
    xt = ax.set_xticklabels(col_labels, rotation=xlabel_rotation, ha="center", fontsize=8)
    ax.set_yticks(range(len(row_order)))
    yt = ax.set_yticklabels(row_order, fontsize=8)

    # Append child counts / leaf marker
    if rowlabel_append_child_counts:
        if groupby not in adata.obs.columns:
            raise KeyError(f"groupby '{groupby}' not found in adata.obs.")
        if rowlabel_child_col not in adata.obs.columns:
            raise KeyError(f"rowlabel_child_col '{rowlabel_child_col}' not found in adata.obs.")

        if rowlabel_child_col == groupby:
            counts = pd.Series(1, index=pd.Index(row_order, name=groupby))
        else:
            tmp = adata.obs[[groupby, rowlabel_child_col]].dropna()
            tmp = tmp.drop_duplicates([groupby, rowlabel_child_col])
            counts = tmp.groupby(groupby, sort=False)[rowlabel_child_col].nunique()

        display_labels = []
        for r in row_order:
            n = int(counts.get(r, 0))
            if n <= 1:
                lbl = f"{r} •"
            else:
                lbl = rowlabel_fmt.format(name=r, n=n)
            display_labels.append(lbl)
        ax.set_yticklabels(display_labels, fontsize=8)

        if left_time_strip:
            # Convert ticklabel gap from points → inches
            lbl_gap_in = _pt(yticklabel_gap_pt)

            # Figure + axis width in inches
            fig_w, fig_h = fig.get_size_inches()
            pos = ax.get_position()
            ax_width_in = fig_w * pos.width if fig_w > 0 else 0.0

            # Convert physical gap (in) → axes fraction
            if ax_width_in > 0:
                x_label_frac = - lbl_gap_in / ax_width_in
            else:
                x_label_frac = 0.0

            # x in axes fraction, y in data → stable rows + constant physical gap
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

            for label in ax.get_yticklabels():
                label.set_ha("right")
                label.set_transform(trans)
                label.set_x(x_label_frac)

    # Clean up lines/grids
    ax.grid(False)
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")
    ax.tick_params(axis="both", which="both", length=0)
    ax.minorticks_off()

    # Title & spines
    if title:
        ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")

    # ----------------- CONSENSUS PANELS (fixed-height sibling axes) -----------------
    def _build_study_palette(studies_order, base="tab10", from_cmap=None, t_range=(0.1, 0.9)):
        S_ = len(studies_order)
        if S_ == 0:
            return []
        if from_cmap is not None:
            cmap_obj_ = plt.get_cmap(from_cmap) if isinstance(from_cmap, str) else from_cmap
            t0, t1 = float(t_range[0]), float(t_range[1])
            ts = np.linspace(t0, t1, S_, endpoint=True)
            return [cmap_obj_(t) for t in ts]
        if S_ <= 10:
            base_cmap = plt.get_cmap(base)
            return [base_cmap(i) for i in range(10)][:S_]
        base_cmap = plt.get_cmap("tab20")
        colors_ = [base_cmap(i) for i in range(20)]
        return [colors_[i % 20] for i in range(S_)]

    def _canon_study_name(s: str) -> str:
        s = re.sub(r"^[\s'\"\[\]]+|[\s'\"\[\]]+$", "", str(s))
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _parse_listlike_strict(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple, set)):
            return [
                _canon_study_name(v)
                for v in x
                if v is not None and str(v).strip().lower() not in ("nan", "none")
            ]
        if hasattr(x, "tolist"):
            return _parse_listlike_strict(x.tolist())
        s = str(x).strip()
        if not s or s.lower() in ("nan", "none"):
            return []
        try:
            lit = ast.literal_eval(s)
            if isinstance(lit, (list, tuple, set)):
                out = [
                    _canon_study_name(v)
                    for v in lit
                    if v is not None and str(v).strip().lower() not in ("nan", "none")
                ]
                return list(dict.fromkeys(out))
        except Exception:
            pass
        toks = re.split(r"[;,\|]+", s)
        out = [
            _canon_study_name(t)
            for t in toks
            if t and _canon_study_name(t) and _canon_study_name(t).lower() not in ("nan", "none")
        ]
        return list(dict.fromkeys(out))

    has_any_support = (support_col in grid.columns and grid[support_col].notna().any())
    if consensus_panel in ("line", "strip") and not has_any_support:
        raise ValueError(
            f"consensus_panel='{consensus_panel}' requires non-null values in support_col='{support_col}'."
        )

    if consensus_panel == "line" and has_any_support:
        if support_line_agg == "median":
            gene_support = grid.groupby("gene", sort=False)[support_col].median()
        elif support_line_agg == "mean":
            gene_support = grid.groupby("gene", sort=False)[support_col].mean()
        else:
            raise ValueError("support_line_agg must be 'median' or 'mean'.")

        yvals = np.array([float(gene_support.get(g, np.nan)) for g in cols_df["gene"]], dtype=float)
        yvals = np.nan_to_num(yvals, nan=0.0)

        line_ax = _add_top_sibling_axes(ax, height_in=consensus_height_in, pad_in=0.0)

        x = np.arange(col_count)
        line_ax.plot(
            x, yvals,
            color=support_line_color,
            linewidth=support_line_lw,
            marker=support_line_marker,
            markersize=support_line_ms,
            alpha=support_line_alpha,
        )
        line_ax.set_xlim(-0.5, col_count - 0.5)
        if support_line_ylim is not None:
            line_ax.set_ylim(*support_line_ylim)
        line_ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        line_ax.tick_params(axis="y", labelleft=False, length=0)
        line_ax.set_xticks([])
        line_ax.grid(False)
        line_ax.xaxis.grid(False, which="both")
        line_ax.yaxis.grid(False, which="both")
        line_ax.spines["top"].set_visible(False)
        line_ax.spines["right"].set_visible(False)
        line_ax.spines["left"].set_linewidth(1)
        line_ax.spines["bottom"].set_linewidth(1)

    elif consensus_panel in ("support_grid", "stacked_bar"):
        sub = design.loc[:, ["celltype", "gene", support_stack_field]].copy()
        sub["__studies__"] = sub[support_stack_field].apply(_parse_listlike_strict)

        studies_order, seen = [], set()
        for g in cols_df["gene"]:
            for lst in sub.loc[sub["gene"] == g, "__studies__"]:
                for st in lst:
                    if st not in seen:
                        seen.add(st)
                        studies_order.append(st)
        S = len(studies_order)
        if S > 0:
            base_choice = "tab10"
            colors = _build_study_palette(
                studies_order,
                base=base_choice if consensus_panel == "support_grid" else base_choice,
                from_cmap=support_palette_from_cmap,
                t_range=support_palette_range,
            )

            # Fixed-height panel above dotplot
            panel_height_in = (
                support_grid_height_in
                if consensus_panel == "support_grid" and support_grid_height_in is not None
                else stacked_bar_height_in
                if consensus_panel == "stacked_bar" and stacked_bar_height_in is not None
                else consensus_height_in
            )
            top_ax = _add_top_sibling_axes(ax, height_in=panel_height_in, pad_in=0.0)

            if consensus_panel == "support_grid":
                M = np.zeros((S, col_count), dtype=int)
                st_to_i = {s: i for i, s in enumerate(studies_order)}
                by_gene = sub.groupby("gene")["__studies__"].apply(list)
                for j, g in enumerate(cols_df["gene"]):
                    present = set()
                    for lst in by_gene.get(g, []):
                        present.update(lst)
                    for st in present:
                        M[st_to_i[st], j] = st_to_i[st] + 1

                listed = ListedColormap([(1, 1, 1, 1)] + colors)
                top_ax.imshow(
                    M,
                    aspect="auto",
                    interpolation="nearest",
                    cmap=listed,
                    vmin=0,
                    vmax=max(1, S),
                    extent=[-0.5, col_count - 0.5, -0.5, S - 0.5],
                )
                top_ax.set_xlim(-0.5, col_count - 0.5)
                top_ax.set_ylim(-0.5, max(S - 0.5, 0.5))
                top_ax.set_xticks([])
                top_ax.set_yticks([])
                top_ax.grid(False)
                for s in top_ax.spines.values():
                    s.set_visible(True)
                    s.set_linewidth(1)

                if support_grid_show_legend:
                    handles = [
                        Patch(facecolor=colors[i], edgecolor="none", label=studies_order[i])
                        for i in range(S)
                    ]
                    top_ax.legend(
                        handles=handles,
                        loc="lower left",
                        bbox_to_anchor=(0, 1.02),
                        ncol=support_grid_legend_ncols,
                        frameon=False,
                        fontsize=8,
                    )

            else:  # "stacked_bar"
                st_to_color = {s: colors[i % len(colors)] for i, s in enumerate(studies_order)}
                by_gene = sub.groupby("gene")["__studies__"].apply(list)

                max_h = 0
                for j, g in enumerate(cols_df["gene"]):
                    present = set()
                    for lst in by_gene.get(g, []):
                        present.update(lst)
                    stack_list = [s for s in studies_order if s in present]
                    for k, st in enumerate(stack_list):
                        rect = mpl.patches.Rectangle(
                            (j - 0.5, k - 0.5), 1.0, 1.0, facecolor=st_to_color[st], edgecolor="none",
                        )
                        top_ax.add_patch(rect)
                    max_h = max(max_h, len(stack_list))

                top_ax.set_xlim(-0.5, col_count - 0.5)
                top_ax.set_ylim(-0.5, max(max_h - 0.5, 0.5))
                top_ax.set_xticks([])
                top_ax.set_yticks([])
                top_ax.grid(False)
                for s in top_ax.spines.values():
                    s.set_visible(True)
                    s.set_linewidth(1)

                if stacked_bar_show_legend:
                    handles = [
                        Patch(facecolor=colors[i], edgecolor="none", label=studies_order[i])
                        for i in range(S)
                    ]
                    top_ax.legend(
                        handles=handles,
                        loc="lower left",
                        bbox_to_anchor=(0, 1.02),
                        ncol=stacked_bar_legend_ncols,
                        frameon=False,
                        fontsize=8,
                    )

    elif consensus_panel == "strip" and has_any_support:
        if support_strip_agg == "median":
            gene_support = grid.groupby("gene", sort=False)[support_col].median()
        elif support_strip_agg == "mean":
            gene_support = grid.groupby("gene", sort=False)[support_col].mean()
        else:
            raise ValueError("support_strip_agg must be 'median' or 'mean'.")

        vals_strip = np.array(
            [float(gene_support.get(g, np.nan)) for g in cols_df["gene"]],
            dtype=float,
        )[None, :]
        vals_strip = np.nan_to_num(vals_strip, nan=0.0)

        svmin = 0.0 if support_strip_vmin is None else support_strip_vmin
        svmax = 1.0 if support_strip_vmax is None else support_strip_vmax

        top_ax = _add_top_sibling_axes(ax, height_in=_pt(strip_height_pt), pad_in=0.0)

        im_strip = top_ax.imshow(
            vals_strip,
            aspect="auto",
            interpolation="nearest",
            cmap=support_strip_cmap,
            vmin=svmin,
            vmax=svmax,
            extent=[-0.5, col_count - 0.5, -0.5, 0.5],
        )
        top_ax.set_xlim(ax.get_xlim())
        top_ax.set_xticks([])
        top_ax.set_yticks([])
        top_ax.grid(False)
        for s in top_ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1)

        # Optional colorbar for strip
        if support_strip_show_colorbar:
            cax_strip = _inset_fixed(
                top_ax, width=_pt(cbar_width_pt), height="80%", loc="center left",
                bbox_to_anchor=(1.02, 0.5, 0, 0), borderpad=0.0
            )
            cbar_strip = plt.colorbar(
                mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(vmin=svmin, vmax=svmax),
                    cmap=support_strip_cmap,
                ),
                cax=cax_strip,
            )
            cbar_strip.ax.tick_params(labelsize=7)

    # --------------------- Color tick labels by group -----------------------
    # Important: ytick texts may include suffixes ("name (n)" or "name •").
    # Color using the *canonical* names from `row_order` by index, not the label text.
    if group_color_dict is not None:
        # Y (cells): map by index -> canonical name
        yticks = ax.get_yticklabels()
        for i, label in enumerate(yticks):
            if 0 <= i < len(row_order):
                ct_base = row_order[i]
                if ct_base in group_color_dict:
                    label.set_color(group_color_dict[ct_base])

        # X (genes): unchanged — still keyed by column owner
        xticks = ax.get_xticklabels()
        for j, lbl in enumerate(xticks):
            owner = cols_df.iloc[j]["col_owner"]
            if owner is not None and owner in group_color_dict:
                lbl.set_color(group_color_dict[owner])


    # --------------------- Colorbar in a fixed-width inset ------------------
    if add_colorbar:
        height_str = cbar_height_frac
        cax = _inset_fixed(
            ax,
            width=_pt(cbar_width_pt),        # inches
            height=height_str,               # percent of dotplot height
            loc="center left",
            bbox_to_anchor=cbar_bbox_to_anchor,
            borderpad=0.0,
        )
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
            cax=cax,
        )

        # Add left-aligned title at the top-left corner of the colorbar axis
        cbar.ax.set_title("")
        cbar.ax.text(
            0.0, 1.1,
            cbar_title,
            ha="left", va="bottom",
            rotation=0,
            fontsize=cbar_title_fontsize,
            transform=cbar.ax.transAxes,
        )

        cbar.ax.tick_params(labelsize=cbar_ticklabel_fontsize)

    # --------------------- Size Legend (inside right gutter) ------------------
    if show_size_legend and len(size_legend_fracs) > 0:
        fracs = np.clip(np.array(size_legend_fracs, dtype=float), 0.0, 1.0)
        fr_scaled = np.sqrt(fracs) if size_mode == "sqrt" else fracs
        sizes = (
            s_min + (s_max - s_min) * (fr_scaled / np.max(fr_scaled))
            if np.max(fr_scaled) > 0
            else np.full_like(fr_scaled, s_min)
        )
        handles = [
            ax.scatter(
                [], [], s=s,
                facecolor=size_legend_facecolor,
                edgecolor=size_legend_edgecolor,
                linewidth=size_legend_edge_lw,
            )
            for s in sizes
        ]
        labels = [size_legend_label_fmt.format(f) for f in fracs]

        legend_kwargs = dict(
            loc=size_legend_loc,
            frameon=False,
            scatterpoints=1,
            handletextpad=0.9,
            columnspacing=1.0,
            labelspacing=0.6,
            borderpad=0.0,
            framealpha=size_legend_framealpha,
            title=size_legend_title,
        )
        if size_legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = size_legend_bbox_to_anchor

        size_legend = ax_leg.legend(
            handles,
            labels,
            **legend_kwargs,
        )
        for text in size_legend.get_texts():
            text.set_fontsize(size_legend_label_fontsize)
        size_legend.get_title().set_fontsize(size_legend_title_fontsize)
        ax_leg.add_artist(size_legend)

    # --------------------- Ring Legend (inside right gutter) ------------------
    if show_ring_legend and add_support_ring and len(ring_legend_fracs) > 0:
        rfracs = np.clip(np.array(ring_legend_fracs, dtype=float), 0.0, 1.0)
        lw_ring = ring_min_lw + (ring_max_lw - ring_min_lw) * rfracs
        base_size = (s_min + s_max) / 2.0
        ring_handles = [
            ax.scatter([], [], s=base_size, facecolors="none",
                      edgecolors=ring_color, linewidths=lw_, alpha=ring_alpha)
            for lw_ in lw_ring
        ]
        ring_labels = [ring_legend_label_fmt.format(f) for f in rfracs]

        legend_kwargs = dict(
            loc=ring_legend_loc,
            frameon=False,
            scatterpoints=1,
            handletextpad=0.9,
            columnspacing=1.0,
            labelspacing=0.6,
            borderpad=0.0,
            framealpha=size_legend_framealpha,
            title=ring_legend_title,
        )
        if ring_legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = ring_legend_bbox_to_anchor

        ring_legend = ax_leg.legend(
            ring_handles,
            ring_labels,
            **legend_kwargs,
        )
        for text in ring_legend.get_texts():
            text.set_fontsize(ring_legend_label_fontsize)
        ring_legend.get_title().set_fontsize(ring_legend_title_fontsize)
        ax_leg.add_artist(ring_legend)

    # --------------------- OPTIONAL left inset time strip -------------------
    if left_time_strip:
        _add_left_time_strip_inset(
            fig,
            ax,
            adata=adata,
            groupby=groupby,
            row_order=row_order,
            time_key=time_key,
            vmin=time_vmin,
            vmax=time_vmax,
            cmap=time_cmap,
            width_pt=time_strip_width_pt,
            gap_pt=time_strip_gap_pt,
            show_yticklabels=time_strip_show_yticklabels,
            label_colors=time_strip_label_colors,
            draw_border=False,
        )

    return ax, grid

# ---------------------------------------------------------------------
# Dotplot Wrapper: Siblings+Tissues vs Consensus Markers 
# ---------------------------------------------------------------------

def group_siblings_vs_markers(
    adata: ad.AnnData,
    node: str,
    *,
    # ontology level
    level_col: str | None = None,
    parent_col: str | None = None,

    # marker selection
    marker_type: Literal["overall", "exclusivity", "contrast", "consensus"] = "overall",
    n_markers: int = 20,
    min_support_ratio: float | None = None,
    min_log2fc: float | None = None,
    min_enrich: float | None = None,
    omit_unannotated: bool = False,

    # expression source
    layer: str | None = "tpm_log",
    use_raw: bool | None = None,

    # style
    highlight_color: str = "black",
    standard_scale: Literal["var", "obs"] | None = None,
    cmap: str = "Blues",
    enforce_global_colorscale: bool = True,

    # layout / sizing
    width_per_gene: float = 0.01,
    height_per_group: float = 0.2,
    min_figsize: tuple[float, float] = (0.5, 0.5),
    max_figsize: tuple[float, float] = (25.0, 25.0),
    hspace: float = 0.04,

    # time strip & consensus
    left_time_strip: bool = False,
    consensus_panel: str | None = None,
    show_size_legend: bool = True,
    xlabel_rotation: int = 90,
    duplicate_gene_columns: bool = False,

    # legends
    show_ring_legend: bool = True,
    add_colorbar: bool = True,

    # titles
    sibling_title: str | None = None,
    tissue_title: str | None = None,
    **dotplot_kwargs,
):
    """
    Single-figure, two-block dotplot:

        Top:   siblings of focal node at level_col
        Bottom:Tissues (ZMAP_Tissue)

    Both blocks share:
      - same columns (marker genes of the focal node)
      - one shared right-hand legend/colorbar gutter.

    Row order (both blocks) is by descending mean expression across marker genes.
    Rings in the bottom panel show 'support_ratio' for any (Tissue, gene) pair
    present in the Tissue-level consensus marker table (filtered to the genes used).
    """
    node_str = str(node)

    # ---- 1) Determine level and parent ----
    if level_col is None:
        level_col = find_level_for_node(adata, node_str)

    if level_col not in _ZMAP_LEVEL_CONFIG:
        raise KeyError(f"level_col={level_col!r} not recognized in _ZMAP_LEVEL_CONFIG.")

    if parent_col is None:
        parent_col = _ZMAP_LEVEL_CONFIG[level_col]["parent_col"]

    # will we have a sibling block?
    has_sibling_block = (level_col != "ZMAP_Tissue")
    parent_label: Optional[str] = None

    # ---- 2) Focal markers and gene set ----
    marker_df = get_node_markers(
        node=node_str,
        level_col=level_col,
        marker_types=(marker_type,),
        n_per_type=n_markers,
        min_support_ratio=min_support_ratio,
        min_log2fc=min_log2fc,
        min_enrich=min_enrich,
        omit_unannotated=omit_unannotated,
    )
    node_markers = marker_df.loc[marker_df["celltype"] == node_str]
    if node_markers.empty:
        raise ValueError(f"No rows in marker_df for focal node={node_str!r}.")
    genes = node_markers["gene"].astype(str).unique().tolist()

    # ---- 3) Siblings (if not Tissue-level) ----
    if has_sibling_block:
        parent_label, siblings = get_parent_and_siblings(
            adata,
            node=node_str,
            level_col=level_col,
            parent_col=parent_col,
        )

        siblings = reorder_groups_by_mean_expression(
            adata,
            groups=siblings,
            level_col=level_col,
            genes=genes,
            layer=layer,
            use_raw=use_raw,
        )
    else:
        siblings = []

    # ---- 4) Tissues block (list of tissues, not yet design_df) ----
    if "ZMAP_Tissue" not in adata.obs.columns:
        raise KeyError("ZMAP_Tissue column not found in adata.obs; required for tissue block.")

    tissues_all = (
        adata.obs["ZMAP_Tissue"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    tissues_all = sorted(tissues_all)

    tissues = reorder_groups_by_mean_expression(
        adata,
        groups=tissues_all,
        level_col="ZMAP_Tissue",
        genes=genes,
        layer=layer,
        use_raw=use_raw,
    )

    # ---- 4b) Reorder genes by global mean (column order) ----
    if has_sibling_block:
        # use siblings as the reference universe at this level
        genes = reorder_genes_by_mean_expression(
            adata,
            genes,
            groups=siblings,
            level_col=level_col,
            layer=layer,
            use_raw=use_raw,
        )
    else:
        # tissue-only case
        genes = reorder_genes_by_mean_expression(
            adata,
            genes,
            groups=tissues,
            level_col="ZMAP_Tissue",
            layer=layer,
            use_raw=use_raw,
        )

    # ---- 5) Now build design_dfs using the reordered genes ----
    if has_sibling_block:
        design_sib = make_sibling_design_df(
            node=node_str,
            siblings=siblings,
            marker_df=marker_df,
            genes=genes,                  # <<< enforce canonical gene order here
            support_col="support_ratio",
        )
        # Optional: filter to genes (redundant but harmless)
        design_sib = design_sib[design_sib["gene"].isin(genes)]
        group_color_sib = {ct: "0.4" for ct in siblings}
        group_color_sib[node_str] = highlight_color

        if sibling_title is None:
            if parent_col is not None and parent_label is not None:
                sibling_title = f"{level_col} = {node_str}\nparent[{parent_col}] = {parent_label}"
            else:
                sibling_title = f"{level_col} = {node_str}"
    else:
        design_sib = None
        group_color_sib = {}

    # tissues design_df with same gene order
    design_tiss = pd.DataFrame(
        [(t, g) for t in tissues for g in genes],
        columns=["celltype", "gene"],
    )

    # ---- 5b) Attach Tissue-level support_ratio for (Tissue, gene) pairs ----
    try:
        tissue_support_df = load_consensus_markers(
            level="Tissue",
            groups=tissues,
            marker_type=marker_type,
            n_per_group=None,
            min_support_ratio=min_support_ratio,
            min_log2fc=min_log2fc,
            min_enrich=min_enrich,
            omit_unannotated=omit_unannotated,
            format="table",
        )
    except TypeError:
        tissue_support_df = load_consensus_markers(
            level="Tissue",
            groups=tissues,
            marker_type=marker_type,
            n_per_group=1000,
            min_support_ratio=min_support_ratio,
            min_log2fc=min_log2fc,
            min_enrich=min_enrich,
            omit_unannotated=omit_unannotated,
            format="table",
        )

    if tissue_support_df is not None and not tissue_support_df.empty:
        ts = tissue_support_df.copy()
        ts["celltype"] = ts["celltype"].astype(str)
        ts["gene"] = ts["gene"].astype(str)
        ts = ts[ts["gene"].isin(genes)]

        if "support_ratio" in ts.columns:
            ts_small = (
                ts
                .drop_duplicates(["celltype", "gene"])
                .loc[:, ["celltype", "gene", "support_ratio"]]
            )
            design_tiss = design_tiss.merge(
                ts_small,
                on=["celltype", "gene"],
                how="left",
            )
        else:
            design_tiss["support_ratio"] = np.nan
    else:
        design_tiss["support_ratio"] = np.nan

    group_color_tiss = {t: "0.4" for t in tissues}
    if node_str in tissues:
        group_color_tiss[node_str] = highlight_color

    if tissue_title is None:
        tissue_title = f"Tissue context for node = {node_str}"

    # ---- 5c) Sanity check: ensure gene order matches between sibling & tissue blocks ----
    if has_sibling_block and design_sib is not None:
        def _ordered_genes(df: pd.DataFrame) -> list[str]:
            # preserve first-occurrence order of 'gene' values
            return list(dict.fromkeys(df["gene"].astype(str).tolist()))

        sib_genes_order = _ordered_genes(design_sib)
        tiss_genes_order = _ordered_genes(design_tiss)

        if sib_genes_order != tiss_genes_order:
            raise RuntimeError(
                "Gene order mismatch between sibling and tissue design_dfs.\n"
                f"sib_genes:  {sib_genes_order}\n"
                f"tiss_genes: {tiss_genes_order}"
            )

    # ---- 6) Figure sizing: keep per-row and per-col spacing consistent ----
    n_cols = len(genes)
    n_rows_sib = len(siblings) if has_sibling_block else 0
    n_rows_tiss = len(tissues)

    total_rows = n_rows_sib + n_rows_tiss
    width = width_per_gene * max(n_cols, 1) + 3.0
    height = height_per_group * max(total_rows, 1) + 2.5

    width = float(np.clip(width, min_figsize[0], max_figsize[0]))
    height = float(np.clip(height, min_figsize[1], max_figsize[1]))

    fig = plt.figure(figsize=(width, height))

    if has_sibling_block:
        gs = fig.add_gridspec(
            2, 1,
            height_ratios=[max(n_rows_sib, 1), max(n_rows_tiss, 1)],
            hspace=hspace,
        )
        ax_sib = fig.add_subplot(gs[0, 0])
        ax_tiss = fig.add_subplot(gs[1, 0], sharex=ax_sib)
    else:
        gs = fig.add_gridspec(1, 1)
        ax_sib = None
        ax_tiss = fig.add_subplot(gs[0, 0])

    # ---- 7) Shared right-hand gutter for legends/colorbar ----
    base = ax_tiss.get_position()
    gutter_left = base.x1 + 0.01
    gutter_right = 1.0 - 0.02
    gutter_width = max(0.01, gutter_right - gutter_left)
    ax_leg = fig.add_axes([gutter_left, base.y0, gutter_width, base.height])
    ax_leg.axis("off")

    # ---- 8) Optional: global colorscale (vmin/vmax) across both panels ----
    vmin_global = vmax_global = None
    if enforce_global_colorscale:
        # compute once from both sibling + tissue designs
        designs = []
        if design_sib is not None:
            designs.append(("sib", design_sib, level_col))
        designs.append(("tiss", design_tiss, "ZMAP_Tissue"))

        all_vals = []
        for _, ddf, gcol in designs:
            _, g_grid = plot_dotplot_design_fullgrid(
                adata,
                design_df=ddf,
                groupby=gcol,
                layer=layer,
                use_raw=use_raw,
                standard_scale=standard_scale,
                cmap=cmap,
                add_colorbar=False,
                show_size_legend=False,
                show_ring_legend=False,
                figsize=(1, 1),
            )
            all_vals.append(g_grid["mean_expr"].to_numpy(dtype=float))

        if all_vals:
            all_vals_concat = np.concatenate(all_vals)
            finite = np.isfinite(all_vals_concat)
            if finite.any():
                vmin_global = np.nanpercentile(all_vals_concat[finite], 1)
                vmax_global = np.nanpercentile(all_vals_concat[finite], 99)

        # close ONLY the temporary figures created above
        for f in plt.get_fignums():
            fig_tmp = plt.figure(f)
            # identify tiny temp figs by size (width < 2in AND height < 2in)
            w, h = fig_tmp.get_size_inches()
            if w < 2 and h < 2:
                plt.close(fig_tmp)

    # ---- 9) Draw sibling block (top) ----
    grid_sib = None
    if has_sibling_block:
        ax_sib, grid_sib = plot_dotplot_design_fullgrid(
            adata,
            design_df=design_sib,
            groupby=level_col,
            layer=layer,
            use_raw=use_raw,
            standard_scale=standard_scale,
            cmap=cmap,
            vmin=vmin_global,
            vmax=vmax_global,
            group_color_dict=group_color_sib,
            consensus_panel=consensus_panel,
            left_time_strip=left_time_strip,
            show_size_legend=False,
            show_ring_legend=False,
            add_colorbar=False,
            xlabel_rotation=xlabel_rotation,
            duplicate_gene_columns=duplicate_gene_columns,
            title=sibling_title,
            figsize=None,
            ax=ax_sib,
            ax_leg=ax_leg,
            adjust_right_gutter=False,
            **dotplot_kwargs,
        )
        ax_sib.tick_params(axis="x", labelbottom=False)

    # ---- 10) Draw tissue block (bottom) ----
    ax_tiss, grid_tiss = plot_dotplot_design_fullgrid(
        adata,
        design_df=design_tiss,
        groupby="ZMAP_Tissue",
        layer=layer,
        use_raw=use_raw,
        standard_scale=standard_scale,
        cmap=cmap,
        vmin=vmin_global,
        vmax=vmax_global,
        group_color_dict=group_color_tiss,
        consensus_panel=consensus_panel,
        left_time_strip=left_time_strip,
        show_size_legend=show_size_legend,
        show_ring_legend=show_ring_legend,
        add_colorbar=add_colorbar,
        xlabel_rotation=xlabel_rotation,
        duplicate_gene_columns=duplicate_gene_columns,
        title='',
        figsize=None,
        ax=ax_tiss,
        ax_leg=ax_leg,
        adjust_right_gutter=False,
        **dotplot_kwargs,
    )

    # ---- 11) Bold the parent tissue row (if applicable) ----
    if parent_label is not None and "ZMAP_Tissue" in adata.obs.columns:
        for lab in ax_tiss.get_yticklabels():
            txt = lab.get_text()
            base_name = txt.split(" (")[0].split(" •")[0]
            if base_name == parent_label:
                lab.set_fontweight("bold")
                break

    # ---- 12) Duplicate gene labels at top ----
    xticks = ax_tiss.get_xticks()
    xlabels = [t.get_text() for t in ax_tiss.get_xticklabels()]

    anchor_ax = ax_sib if ax_sib is not None else ax_tiss
    top_ax = anchor_ax.secondary_xaxis("top")
    top_ax.set_xticks(xticks)
    top_ax.set_xticklabels(
        xlabels,
        rotation=90,
        ha="center",
        fontsize=8,
    )
    top_ax.tick_params(axis="x", which="both", length=0)
    top_ax.set_xlabel("")

    # -----------------------------------------------------------
    # Centered title for ZMAP_Tissue nodes (single-panel version)
    # -----------------------------------------------------------
    if level_col == "ZMAP_Tissue":
        anchor_ax = ax_sib if ax_sib is not None else ax_tiss

        if anchor_ax is not None:
            # 1. Measure the xtick label height (in inches)
            fig = anchor_ax.figure
            fig.canvas.draw()  # needed so the renderer exists
            renderer = fig.canvas.get_renderer()

            # get bounding boxes of all xtick labels
            bboxes = [lbl.get_window_extent(renderer=renderer) for lbl in anchor_ax.get_xticklabels()]
            if len(bboxes) > 0:
                # convert from display coords to inches
                max_label_height_px = max(bb.height for bb in bboxes)
                dpi = fig.dpi
                max_label_height_in = max_label_height_px / dpi
            else:
                max_label_height_in = 0.0

            # 2. Add some extra cushion (in inches)
            extra_gap_in = 0.05  # minimal headroom
            required_pad = max_label_height_in + extra_gap_in

            # 3. Create a sibling axes ABOVE the dotplot, padded enough
            title_ax = _add_top_sibling_axes(
                anchor_ax,
                height_in=0.28,      # height of the title band
                pad_in=required_pad  # auto-lift by measured xtick height
            )
            title_ax.set_axis_off()

            # 4. Center the title horizontally and vertically
            title_ax.text(
                0.5, 0.55,                   # centered within the band
                f"{level_col} = {node}",     # displayed title
                fontsize=10,
                ha="center",
                va="center",
            )




    return fig, ax_sib, ax_tiss, (grid_sib, grid_tiss), marker_df

# ---------------------------------------------------------------------
# Dotplot Wrapper: Descendants vs Consensus Markers 
# ---------------------------------------------------------------------

def group_descendants_vs_markers(
    adata: ad.AnnData,
    parent: Optional[str] = None,                     # NEW: the selected parent group (e.g., a Tissue or CellType)
    *,
    # ---- Hierarchy columns (parent/child) ----
    parent_col: str = "ZMAP_CellType",                # NEW: higher-level obs column (e.g., ZMAP_Tissue or ZMAP_CellType)
    child_col: str = "ZMAP_Cluster",                  # NEW: lower-level obs column used for rows (e.g., ZMAP_Cluster or leiden_100)

    # ---- Back-compat aliases (deprecated) ----
    celltype: Optional[str] = None,                   # was the selected value at the parent level
    celltype_col: Optional[str] = None,
    cluster_col: Optional[str] = None,

    # ---- expression source ----
    layer: Optional[str] = "tpm_log",
    use_raw: Optional[bool] = None,

    # ---- marker loading ----
    marker_type: Literal["exclusivity", "contrast", "consensus", "overall"] = "exclusivity",
    n_markers_per_group: Optional[int] = 5,
    min_support_ratio: Optional[float] = None,
    min_log2fc: Optional[float] = None,
    min_enrich: Optional[float] = None,
    omit_unannotated: bool = False,

    # ---- group filtering ----
    min_cells_per_group: Optional[int] = None,

    # ---- ordering / dendrogram ----
    use_dendrogram_rows: bool = False,
    dendrogram_metric: str = "correlation",
    dendrogram_method: str = "average",

    # ---- dotplot scaling & style ----
    standard_scale: Optional[str] = "var",
    detect_threshold: float = 0.0,
    cmap: str = "Blues",
    consensus_panel: Optional[str] = None,
    stacked_bar_show_legend: bool = True,
    group_color_dict: Optional[Mapping[str, str]] = None,
    duplicate_gene_columns: bool = True,

    # ---- figure size control ----
    figsize: Optional[Tuple[float, float]] = None,
    min_figsize: Tuple[float, float] = (9.0, 4.5),
    max_figsize: Tuple[float, float] = (80.0, 40.0),
    width_per_gene: float = 0.15,
    height_per_group: float = 0.15,
    left_margin: float = 1.5,
    right_margin: float = 2.5,
    top_margin: float = 1.0,
    bottom_margin: float = 1.0,

    # ---- misc dotplot passthrough ----
    s_min: float = 1.0,
    s_max: float = 80.0,
    xlabel_rotation: int = 90,
    title: Optional[str] = None,
    show_size_legend: bool = True,
    **dotplot_kwargs,
):
    """
    ZMAP browser for any two hierarchical levels (parent → child).

    Typical examples:
      - parent_col="ZMAP_Tissue",  child_col="ZMAP_Cluster", parent="forebrain"
      - parent_col="ZMAP_CellType", child_col="leiden_100",   parent="Neurons"
    """
    # -------------------- 0) Back-compat parameter resolution --------------------
    if celltype is not None and parent is None:
        warnings.warn("`celltype` is deprecated; use `parent`.", DeprecationWarning, stacklevel=2)
        parent = celltype
    if celltype_col is not None:
        warnings.warn("`celltype_col` is deprecated; use `parent_col`.", DeprecationWarning, stacklevel=2)
        parent_col = celltype_col
    if cluster_col is not None:
        warnings.warn("`cluster_col` is deprecated; use `child_col`.", DeprecationWarning, stacklevel=2)
        child_col = cluster_col

    # -------------------- 1) subset adata to chosen parent ----------------------
    if parent_col not in adata.obs.columns:
        raise KeyError(f"{parent_col!r} not found in adata.obs columns.")
    if parent is None:
        raise ValueError("`parent` must be provided (e.g., a specific Tissue or CellType label).")

    mask = adata.obs[parent_col].astype(str) == str(parent)
    if mask.sum() == 0:
        raise ValueError(f"No cells found with {parent_col} == {parent!r}.")

    # Copy small subset to avoid chained-assignment surprises
    adata_tmp = adata[mask].copy()

    # -------------------- 2) find child groups & optional filtering -------------
    if child_col not in adata_tmp.obs.columns:
        raise KeyError(f"{child_col!r} not found in adata.obs columns.")

    children = adata_tmp.obs[child_col].astype(str)
    group_counts = children.value_counts(sort=False)

    if min_cells_per_group is not None:
        keep_groups = group_counts[group_counts >= int(min_cells_per_group)].index.astype(str).tolist()
    else:
        keep_groups = group_counts.index.astype(str).tolist()

    if len(keep_groups) == 0:
        raise ValueError(
            f"After applying min_cells_per_group={min_cells_per_group}, "
            f"no {child_col} groups remain within parent {parent!r}."
        )

    # Restrict to kept child groups (within this parent)
    adata_tmp = adata_tmp[adata_tmp.obs[child_col].astype(str).isin(keep_groups)].copy()
    children = adata_tmp.obs[child_col].astype(str)
    group_order = list(pd.unique(children))  # preserve observed order

    # -------------------- 3) load consensus markers at child level --------------
    consensus_level = _ZMAP_LEVEL_LOOKUP.get(child_col)
    if consensus_level is None:
        raise KeyError(
            f"child_col={child_col!r} has no mapping in _ZMAP_LEVEL_LOOKUP; "
            f"add it to map to a canonical consensus level."
        )

    marker_df = load_consensus_markers(
        level=consensus_level,
        groups=group_order,                     # restrict at load time
        marker_type=marker_type,
        n_per_group=n_markers_per_group,
        min_support_ratio=min_support_ratio,
        min_log2fc=min_log2fc,
        min_enrich=min_enrich,
        omit_unannotated=omit_unannotated,
        format="table",
    )
    if marker_df.empty:
        raise ValueError(
            f"No markers returned for level={consensus_level!r}, "
            f"marker_type={marker_type!r}, groups={group_order}."
        )

    # Ensure consistent string types
    marker_df["celltype"] = marker_df["celltype"].astype(str)
    marker_df["gene"] = marker_df["gene"].astype(str)

    # Keep only markers for groups actually present
    marker_df = marker_df[marker_df["celltype"].isin(group_order)]
    if marker_df.empty:
        raise ValueError(
            "Marker table is empty after restricting to present groups. "
            "Check that your consensus tables and obs labels match."
        )

    # Drop genes not in var_names
    var_names = adata_tmp.var_names.astype(str)
    present_genes = pd.Index(marker_df["gene"].unique()).intersection(var_names)
    marker_df = marker_df[marker_df["gene"].isin(present_genes)]
    if marker_df.empty:
        raise ValueError("No marker genes overlap with adata.var_names for this subset.")

    # Ordered categorical for consistent row sort
    marker_df["celltype"] = pd.Categorical(marker_df["celltype"], categories=group_order, ordered=True)
    marker_df = marker_df.sort_values(["celltype"]).reset_index(drop=True)

    # -------------------- 4) optional row dendrogram on mean expression ---------
    if use_dendrogram_rows:
        genes_unique = pd.Index(marker_df["gene"].unique())
        gene_idx = var_names.get_indexer(genes_unique)
        valid_mask = gene_idx >= 0
        genes_unique = genes_unique[valid_mask]
        gene_idx = gene_idx[valid_mask]

        if genes_unique.empty:
            raise ValueError("No valid genes remain for dendrogram computation.")

        # choose expression matrix
        if use_raw and (adata_tmp.raw is not None):
            X = adata_tmp.raw.X[:, gene_idx]
        elif layer is not None:
            X = adata_tmp.layers[layer][:, gene_idx]
        else:
            X = adata_tmp.X[:, gene_idx]

        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        # group means
        child_labels = adata_tmp.obs[child_col].astype(str).values
        group_to_idx = {g: np.where(child_labels == g)[0] for g in group_order}
        mat = np.zeros((len(group_order), len(genes_unique)), dtype=float)
        for i, g in enumerate(group_order):
            idx = group_to_idx[g]
            if idx.size > 0:
                sub = X_dense[idx, :]
                mat[i, :] = np.asarray(sub.mean(axis=0)).ravel()

        # If constant rows, keep original order
        if np.allclose(mat, mat[0]):
            new_group_order = group_order
        else:
            d = pdist(mat, metric=dendrogram_metric)
            Z = linkage(d, method=dendrogram_method)
            leaf_idx = leaves_list(Z)
            new_group_order = [group_order[i] for i in leaf_idx]

        group_order = new_group_order
        marker_df["celltype"] = marker_df["celltype"].cat.set_categories(group_order, ordered=True)
        marker_df = marker_df.sort_values(["celltype"]).reset_index(drop=True)

    # -------------------- 5) auto figure size if needed -------------------------
    if figsize is None:
        n_groups = len(group_order)

        # Use the *final* number of columns implied by marker_df
        if duplicate_gene_columns:
            # columns can be repeated per (celltype, gene)
            n_cols = marker_df.drop_duplicates(["celltype", "gene"]).shape[0]
        else:
            # each gene appears only once as a column
            n_cols = marker_df["gene"].nunique()

        width = left_margin + width_per_gene * max(n_cols, 1) + right_margin
        height = top_margin + height_per_group * max(n_groups, 1) + bottom_margin

        width = float(np.clip(width, min_figsize[0], max_figsize[0]))
        height = float(np.clip(height, min_figsize[1], max_figsize[1]))
        figsize = (width, height)

    # -------------------- 6) call the core dotplot function ---------------------
    ax, grid = plot_dotplot_design_fullgrid(
        adata_tmp,
        design_df=marker_df.rename(columns={"celltype": "celltype", "gene": "gene"}),
        groupby=child_col,                          # rows = child level
        layer=layer,
        use_raw=use_raw,
        detect_threshold=detect_threshold,
        cmap=cmap,
        standard_scale=standard_scale,
        s_min=s_min,
        s_max=s_max,
        figsize=figsize,
        xlabel_rotation=xlabel_rotation,
        title=title,
        consensus_panel=consensus_panel,
        stacked_bar_show_legend=stacked_bar_show_legend,
        group_color_dict=None if group_color_dict is None else dict(group_color_dict),
        duplicate_gene_columns=duplicate_gene_columns,
        show_size_legend=show_size_legend,
        **dotplot_kwargs,
    )

    return ax, grid, marker_df
