import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ---------------------------------------------------------------------
# Group orderings & colors
# ---------------------------------------------------------------------

DEFAULT_CLUSTER_ORDER_ZMAP_CellType = [    
    'blastomeres', 
    'ectoderm_progenitor', 'fin_epidermis', 'basal_epidermis', 'ionocytes', 'lens', 'lateral_line', 'olfactory', 'otic', 'periderm', 'superficial_epidermis',
    'endoderm_progenitor', 'pharyngeal_pouch', 'forerunner_cells', 'intestinal_epithelium', 'hepatocyte', 'endocrine_pancreas', 'exocrine_pancreas', 
    'neural_crest_progenitor','pharyngeal_arch_crest', 'iridophore', 'melanocyte', 'xanthophore', 'schwann', 
    'neural_progenitor', 'telencephalon', 'diencephalon', 'mesencephalon', 'rhombocephalon', 'spinal_cord_progenitor', 'differentiating_neurons', 
    'optic_cup', 'retinal_pigment_epithelium', 'retina', 'photoreceptors',
    'mesoderm_progenitor', 'tailbud', 'presomitic_meso', 'adaxial', 'myotome', 'slow_muscle', 'fast_muscle', 'somites', 'dermatome', 'dermis', 'fibroblasts', 'sclerotome', 'axial_skeleton', 'tendon',
    'lateral_plate_progenitor', 'pharyngeal_arch_meso', 'head_mesenchyme', 'meninges', 'periocular_meso', 'fin_bud_mesenchyme', 'fin_bud_chondrocytes', 'facial_skeleton', 'cephalic_muscle', 'visceral_smooth_muscle', 'vascular_smooth_muscle', 'cardiac_mesenchyme', 'cardiac_muscle', 'pronephros', 
    'vasculature', 'vein', 'endocardial', 'aorta', 'lymphatic', 
    'hematopoietic_progenitor', 'erythroid', 'myeloid_progenitor', 'macrophage', 'neutrophil', 
    'notochord', 'prechordal_plate', 'hatching_gland', 
    'pgc_progenitor']

DEFAULT_ROW_COLOR_GROUPS_ZMAP_CellType = {
    "black":      ["blastomeres"],
    "green":      ['ectoderm_progenitor', 'fin_epidermis', 'basal_epidermis', 'ionocytes', 'lens', 'lateral_line', 'olfactory', 'otic', 'periderm', 'superficial_epidermis'],
    "darkorange": ['endoderm_progenitor', 'pharyngeal_pouch', 'forerunner_cells', 'intestinal_epithelium', 'hepatocyte', 'endocrine_pancreas', 'exocrine_pancreas'],
    "steelblue":  ['neural_crest_progenitor','pharyngeal_arch_crest', 'iridophore', 'melanocyte', 'xanthophore', 'schwann'],
    "blue":       ['neural_progenitor', 'telencephalon', 'diencephalon', 'mesencephalon', 'rhombocephalon', 'spinal_cord_progenitor', 'differentiating_neurons'],
    "darkblue":   ['optic_cup', 'retinal_pigment_epithelium', 'retina', 'photoreceptors'],
    "red":        ['mesoderm_progenitor', 'tailbud', 'presomitic_meso', 'adaxial', 'myotome', 'slow_muscle', 'fast_muscle', 'somites', 'dermatome', 'dermis', 'fibroblasts', 'sclerotome', 'axial_skeleton', 'tendon'],
    "crimson":    ['lateral_plate_progenitor', 'pharyngeal_arch_meso', 'head_mesenchyme', 'meninges', 'periocular_meso', 'fin_bud_mesenchyme', 'fin_bud_chondrocytes', 'facial_skeleton', 'cephalic_muscle', 'visceral_smooth_muscle', 'vascular_smooth_muscle', 'cardiac_mesenchyme', 'cardiac_muscle', 'pronephros'],
    "orangered":  ['vasculature', 'vein', 'endocardial', 'aorta', 'lymphatic'],
    "firebrick":  ['hematopoietic_progenitor', 'erythroid', 'myeloid_progenitor', 'macrophage', 'neutrophil'],
    "r":          ['notochord', 'prechordal_plate', 'hatching_gland', ],
    "purple":     ["pgc_progenitor"],
}

DEFAULT_ROW_DIVIDERS_ZMAP_CellType = [1, 11, 18, 24, 31, 35, 49, 63, 68, 73, 76]

DEFAULT_STUDY_ORDER = ["Kamimoto2023", "Farrell2018", "Kukreja2024", "Wagner2018",
"Farnsworth2020", "Lange2023", "Sur2023", "Spanjaard2018"]


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _select_matrix_and_var_names(adata, layer, use_raw):
    """Return (X, var_names) based on raw/layer/X selection."""
    if use_raw and getattr(adata, "raw", None) is not None:
        X = adata.raw.X
        var_names = pd.Index(adata.raw.var_names).astype(str)
    elif layer is not None:
        X = adata.layers[layer]
        var_names = pd.Index(adata.var_names).astype(str)
    else:
        X = adata.X
        var_names = pd.Index(adata.var_names).astype(str)
    return X, var_names


def _extract_gene_vector(X, var_names, gene: str) -> np.ndarray:
    """Return 1D expression vector for a given gene name from X."""
    gene = str(gene)
    if gene not in var_names:
        raise ValueError("Gene %r not found in var_names." % gene)
    j = int(np.where(var_names == gene)[0][0])

    # Handle dense vs sparse
    if hasattr(X, "getcol"):  # scipy.sparse
        col = X.getcol(j).toarray().ravel()
    elif hasattr(X, "tocsc") or hasattr(X, "tocsr"):
        col = np.asarray(X[:, j].toarray()).ravel()
    else:
        col = np.asarray(X[:, j]).ravel()
    return col


def _extract_color_vector(
    adata,
    color: str,
    layer: str | None,
    use_raw: bool | None,
) -> tuple[np.ndarray, str]:
    """
    Resolve `color` (typically a gene) to a per-cell vector.

    Resolution order:
    1) Gene in X/raw/layer (via `_select_matrix_and_var_names`).
    2) Numeric column in `adata.obs`.

    Parameters
    ----------
    adata
        AnnData object.
    color
        Key describing what to plot. Usually a gene name, but can also
        be a numeric `adata.obs` column.
    layer, use_raw
        Same semantics as in the dotplot functions.

    Returns
    -------
    vec : np.ndarray
        1D array of length n_obs with values used for color & size.
    source : {"gene", "obs"}
        Where the values came from.
    """
    from pandas.api.types import is_numeric_dtype

    color = str(color)

    # Try gene in X/raw/layer
    X, var_names = _select_matrix_and_var_names(adata, layer, use_raw)
    if color in var_names:
        vec = _extract_gene_vector(X, var_names, color)
        return vec, "gene"

    # Try numeric obs column
    if color in adata.obs.columns:
        s = adata.obs[color]
        if not is_numeric_dtype(s):
            raise ValueError(
                f"Color key '{color}' found in adata.obs, but dtype is not numeric."
            )
        vec = pd.to_numeric(s, errors="coerce").to_numpy()
        return vec, "obs"

    raise ValueError(
        f"Color key '{color}' not found as gene (var_names/raw/layer) or numeric adata.obs column."
    )


def _scale01(vec: np.ndarray) -> np.ndarray:
    """
    In-place scale of finite entries in `vec` to [0, 1].

    Non-finite values are left untouched. If the dynamic range is zero,
    all finite entries are set to 0.0.
    """
    mask = np.isfinite(vec)
    if not np.any(mask):
        return vec
    v = vec[mask]
    vmin_ = np.nanmin(v)
    vmax_ = np.nanmax(v)
    if (not np.isfinite(vmin_)) or (not np.isfinite(vmax_)) or vmax_ == vmin_:
        vec[mask] = 0.0
    else:
        vec[mask] = (vec[mask] - vmin_) / (vmax_ - vmin_)
    return vec


def _compute_sizes_from_fraction(
    frac: np.ndarray,
    s_min: float,
    s_max: float,
    size_zero_for_missing: bool,
) -> np.ndarray:
    """Map fraction-expressing values to pixel sizes."""
    if size_zero_for_missing:
        frac_plot = np.nan_to_num(frac, nan=0.0)
    else:
        frac_plot = frac
    f = np.clip(frac_plot, 0.0, 1.0)
    sizes = s_min + f * (s_max - s_min)
    sizes[~np.isfinite(sizes)] = s_min
    return sizes


def _compute_color_norm(vals_color, vmin, vmax):
    """
    Compute a matplotlib.Normalize for given color values and vmin/vmax behavior.

    If vmin/vmax are None, percentiles (1, 99) are used on finite values.
    """
    finite_vals = vals_color[np.isfinite(vals_color)]
    if vmin is not None:
        _vmin = vmin
    else:
        _vmin = np.nanpercentile(finite_vals, 1) if finite_vals.size else 0.0
    if vmax is not None:
        _vmax = vmax
    else:
        _vmax = np.nanpercentile(finite_vals, 99) if finite_vals.size else 1.0
    norm = mpl.colors.Normalize(vmin=_vmin, vmax=_vmax)
    return norm, _vmin, _vmax


def _style_axes_spines(ax, lw: float = 0.5):
    """Turn off ticks grid, keep spines thin but visible."""
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)


def _bold_first_rows(ax, row_dividers):
    """Bold the first ytick in each row group defined by row_dividers."""
    if not row_dividers:
        return
    group_starts = [0] + list(row_dividers)
    yticklabels = ax.get_yticklabels()
    for idx in group_starts:
        if 0 <= idx < len(yticklabels):
            yticklabels[idx].set_fontweight("bold")


def _color_row_labels(ax, row_color_groups):
    """
    Color ytick labels according to row_color_groups mapping:
    {color: [cluster1, cluster2, ...]}.
    """
    if not row_color_groups:
        return
    name_to_color = {
        name: col for col, names in row_color_groups.items() for name in names
    }
    for label in ax.get_yticklabels():
        txt = label.get_text()
        if txt in name_to_color:
            label.set_color(name_to_color[txt])


def _draw_tick_grids(
    ax,
    n_x: int,
    n_y: int,
    tick_grid_x: bool,
    tick_grid_y: bool,
    tick_grid_lw: float,
    tick_grid_alpha: float,
):
    """Draw faint vertical/horizontal grid lines at integer positions."""
    if tick_grid_x:
        for xloc in range(n_x):
            ax.axvline(
                xloc,
                color="0.85",
                lw=tick_grid_lw,
                alpha=tick_grid_alpha,
                zorder=0,
            )
    if tick_grid_y:
        for yloc in range(n_y):
            ax.axhline(
                yloc,
                color="0.85",
                lw=tick_grid_lw,
                alpha=tick_grid_alpha,
                zorder=0,
            )


def _draw_row_dividers(
    ax,
    C: int,
    row_dividers,
    row_divider_color: str,
    row_divider_lw: float,
    row_divider_alpha: float,
):
    """Draw horizontal lines between rows at indices listed in row_dividers."""
    if not row_dividers:
        return
    for idx in row_dividers:
        try:
            idx = int(idx)
        except Exception:
            continue
        if 0 <= idx <= C - 1:
            y = idx - 0.5
            ax.axhline(
                y,
                color=row_divider_color,
                lw=row_divider_lw,
                alpha=row_divider_alpha,
                zorder=1,
            )


def _add_vertical_colorbar(
    fig,
    norm,
    cmap_obj,
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
    title: str,
    tick_fontsize: int = 8,
    title_fontsize: int = 8,
):
    """
    Add a vertically oriented colorbar to the figure at figure-fraction coords.

    Returns the colorbar axis and Colorbar object.
    """
    cax = fig.add_axes([left, bottom, width, height])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.ax.set_title(title, fontsize=title_fontsize, pad=6)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    return cax, cbar


def _add_fraction_size_legend(
    fig,
    *,
    s_min: float,
    s_max: float,
    size_legend_values,
    size_legend_title: str,
    size_legend_facecolor: str,
    left: float,
    bottom: float,
    width: float,
    height: float,
    background: str,
):
    """
    Add a legend showing mapping from fraction-expressing to dot size.

    size_legend_values: iterable of fractions in [0, 1].
    """
    base_vals = np.asarray(
        size_legend_values if size_legend_values is not None
        else (0.1, 0.25, 0.5, 0.75, 1.0),
        dtype=float,
    )
    base_vals = np.clip(base_vals, 0.0, 1.0)

    legend_sizes = s_min + base_vals * (s_max - s_min)
    order = np.argsort(base_vals)[::-1]
    y_positions = (np.arange(len(order)) * 0.5)[::-1]

    leg_ax = fig.add_axes([left, bottom, width, height], facecolor=background)
    leg_ax.set_in_layout(False)

    for i, idx in enumerate(order):
        s_ = legend_sizes[idx]
        lab_val = base_vals[idx]
        leg_ax.scatter(
            [0.2],
            [y_positions[i]],
            s=s_,
            facecolor=size_legend_facecolor,
            edgecolor="none",
            linewidth=0,
            zorder=3,
            clip_on=False,
        )
        label = f"{lab_val * 100:.0f}%         "
        leg_ax.text(
            0.45,
            y_positions[i],
            label,
            va="center",
            ha="left",
            fontsize=8,
            clip_on=False,
        )

    leg_ax.set_xlim(0.0, 1.0)
    if len(order):
        leg_ax.set_ylim(-0.5, y_positions.max() + 0.7)
    else:
        leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")

    if size_legend_title is None:
        size_legend_title = "Fraction\nExpressing"

    leg_ax.text(
        0.02,
        (y_positions.max() + 0.6) if len(order) else 0.6,
        size_legend_title,
        fontsize=8,
        va="bottom",
        ha="left",
        rotation=0,
        clip_on=False,
    )


def _add_row_color_legend(
    fig,
    cluster_order,
    row_color_groups,
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
    background: str,
):
    """
    Add a legend summarizing row color groups.

    Shows one short colored line per group, labelled by color and count.
    """
    if not row_color_groups:
        return

    counts_by_color = {
        col: sum(1 for n in names if n in cluster_order)
        for col, names in row_color_groups.items()
    }

    sw_ax = fig.add_axes([left, bottom, width, height], facecolor=background)
    y0 = 0.9
    dy = 0.18
    for i, (col, count) in enumerate(
        sorted(counts_by_color.items(), key=lambda kv: (-kv[1], str(kv[0])))
    ):
        y = y0 - i * dy
        sw_ax.plot(
            [0.05, 0.20],
            [y, y],
            color=col,
            lw=6,
            solid_capstyle="butt",
        )
        sw_ax.text(
            0.25,
            y,
            f"{col} ({count})",
            va="center",
            ha="left",
            fontsize=8,
            clip_on=False,
        )
    sw_ax.set_xlim(0, 1)
    sw_ax.set_ylim(0, 1)
    sw_ax.axis("off")


# ---------------------------------------------------------------------
# 1) Dotplot: color feature vs time
# ---------------------------------------------------------------------

def gene_groups_vs_time(
    adata,
    color: str,
    *,
    groupby: str = "ZMAP_CellType",
    time_col: str = "time_block_id",
    layer: str | None = "tpm_log",
    use_raw: bool | None = None,
    detect_threshold: float = 0.0,
    show: bool = True,
    # ===== color (mean expression) =====
    cmap: str = "viridis",
    vmin: float | None = 0,
    vmax: float | None = None,
    standard_scale: str | None = None,    # None | "time" | "cluster"
    # ===== DOT SIZE (fraction expressing) =====
    s_min: float = 4.0,
    s_max: float = 60.0,
    size_zero_for_missing: bool = True,
    # ===== LOW-SUPPORT / MISSING  =====
    abs_min_cells: int = 10,
    rel_min_frac: float = 0.01,
    rel_abs_cap: int = 300,
    low_support_grey: str = "0.7",
    low_support_alpha: float = 0.5,
    # ===== layout/labels =====
    base_col_width: float = 0.14,
    base_row_height: float = 0.14,
    gutter_width: float = 1.5,            # fixed right gutter (plot area is stable)
    gutter_pad: float = 0.03,             # extra right-side pad (in figure coords)
    figsize: tuple | None = None,
    xlabel_rotation: int = 90,
    title: str | None = None,
    add_colorbar: bool = True,
    cbar_title: str = "log(tpm)\ncounts",
    # ===== size legend =====
    show_size_legend: bool = True,
    size_legend_values: tuple | None = None,
    size_legend_title: str | None = None,
    size_legend_facecolor: str = "black",
    # ===== ordering & filtering =====
    cluster_order: list[str] | None = DEFAULT_CLUSTER_ORDER_ZMAP_CellType,
    time_order: list[str] | None = None,
    omit_groups: list[str] | None = ['nan', 'unknown'],
    # ===== cosmetics =====
    edgecolor: str = "black",
    edge_lw: float = 0.1,
    background: str = "white",
    # ===== faint tick/grid lines =====
    tick_grid_x: bool = False,
    tick_grid_y: bool = False,
    tick_grid_alpha: float = 0.15,
    tick_grid_lw: float = 0.6,
    # ===== Row color group mapping & legend =====
    row_color_groups: dict[str, list[str]] | None = DEFAULT_ROW_COLOR_GROUPS_ZMAP_CellType,
    show_row_color_legend: bool = False,
    # ===== Horizontal split lines between rows =====
    row_dividers: list[int] | None = DEFAULT_ROW_DIVIDERS_ZMAP_CellType,
    row_divider_color: str = "black",
    row_divider_lw: float = 0.5,
    row_divider_alpha: float = 1,
    # ===== performance/output controls =====
    return_long: bool = False,
):
    """
    Stable-layout dotplot of a single color feature across (cluster × timepoint).

    `color` is usually a gene name (taken from raw/layer/X), but if no gene
    match is found it will fall back to a numeric column in `adata.obs`.

    Dot semantics
    -------------
    * Color: mean value per (cluster, time) bin.
    * Size:  fraction of cells with value > `detect_threshold`.
    * Low-support / missing: grey at minimum size.
    """
    color = str(color)

    # ------------------------ color vector ------------------------
    x, color_source = _extract_color_vector(adata, color, layer, use_raw)

    # ------------------------ metadata as arrays (no full copy) --------------
    obs = adata.obs
    if groupby not in obs or time_col not in obs:
        raise ValueError(f"Missing columns in adata.obs: need '{groupby}' and '{time_col}'.")

    clusters_ser = obs[groupby].astype(str)
    times_ser    = obs[time_col].astype(str)

    # Optional omit
    if omit_groups:
        omit_set = set(map(str, omit_groups))
        keep_mask = ~clusters_ser.isin(omit_set)
        clusters_ser = clusters_ser[keep_mask]
        times_ser    = times_ser[keep_mask]
        x            = x[keep_mask.values]
    else:
        omit_set = set()

    # Determine orders (respect categorical if present)
    if cluster_order is None:
        if isinstance(obs[groupby].dtype, pd.CategoricalDtype):
            cluster_order = [
                c for c in obs[groupby].cat.categories.astype(str) if c not in omit_set
            ]
        else:
            cluster_order = list(pd.unique(clusters_ser))
    else:
        present = set(pd.unique(clusters_ser))
        cluster_order = [c for c in cluster_order if c in present]

    if time_order is None:
        if isinstance(obs[time_col].dtype, pd.CategoricalDtype):
            time_order = list(obs[time_col].cat.categories.astype(str))
        else:
            time_order = list(pd.unique(times_ser))

    C = len(cluster_order)
    T = len(time_order)

    # Map to categorical codes aligned to the chosen orders
    clust_cat = pd.Categorical(clusters_ser, categories=cluster_order, ordered=True)
    time_cat  = pd.Categorical(times_ser,    categories=time_order,  ordered=True)
    c_codes = clust_cat.codes.astype(np.int64)
    t_codes = time_cat.codes.astype(np.int64)

    valid = (c_codes >= 0) & (t_codes >= 0)
    c_codes = c_codes[valid]
    t_codes = t_codes[valid]
    x       = x[valid]

    # Linearize (cluster,time) → idx
    lin = (c_codes * T + t_codes).astype(np.int64)

    # ------------------------ fast aggregation via bincount ------------------
    n = np.bincount(lin, minlength=C*T).astype(np.int32)
    is_expr = (x > detect_threshold)
    k = np.bincount(lin, weights=is_expr.astype(np.int8), minlength=C*T).astype(np.int32)
    sum_expr = np.bincount(lin, weights=x, minlength=C*T)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_expr = np.where(n > 0, sum_expr / n, np.nan)

    # N_clust per cluster, N_time per time
    N_clust = np.bincount(c_codes, minlength=C).astype(np.int32)
    N_time  = np.bincount(t_codes, minlength=T).astype(np.int32)

    # ------------------------ low-support eligibility ------------------------
    req = np.maximum(
        abs_min_cells,
        np.minimum(np.ceil(N_clust * float(rel_min_frac)).astype(int), int(rel_abs_cap)),
    ).astype(np.int32)
    req_grid = np.repeat(req[:, None], T, axis=1).ravel()

    eligible = (n >= req_grid)
    has_cells = (n > 0)

    # ------------------------ standard scaling (optional) --------------------
    vals_color = mean_expr.copy()

    if standard_scale == "time":
        vals_color = vals_color.reshape(C, T)
        for ci in range(C):
            vals_color[ci, :] = _scale01(vals_color[ci, :])
        vals_color = vals_color.ravel()
    elif standard_scale == "cluster":
        vals_color = vals_color.reshape(C, T)
        for ti in range(T):
            vals_color[:, ti] = _scale01(vals_color[:, ti])
        vals_color = vals_color.ravel()
    elif standard_scale is not None:
        raise ValueError("standard_scale must be None, 'time', or 'cluster'.")

    # ------------------------ color mapping ------------------------
    cmap_obj = plt.get_cmap(cmap)
    norm, _vmin, _vmax = _compute_color_norm(vals_color, vmin, vmax)

    colors = np.zeros((C*T, 4), dtype=float)
    valid_color = has_cells
    colors[valid_color] = cmap_obj(
        norm(np.nan_to_num(vals_color[valid_color], nan=_vmin))
    )

    # ------------------------ dot sizes (fraction expressing) ----------------
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(n > 0, k / np.maximum(n, 1), np.nan)

    sizes = _compute_sizes_from_fraction(
        frac, s_min=s_min, s_max=s_max, size_zero_for_missing=size_zero_for_missing
    )

    # Treat low-support and missing identically
    low_or_missing = (~eligible) | (~has_cells)
    if np.any(low_or_missing):
        colors[low_or_missing] = mpl.colors.to_rgba(low_support_grey, alpha=low_support_alpha)
        sizes[low_or_missing] = s_min

    # ------------------------ positions ---------------------------
    xs = np.tile(np.arange(T, dtype=float), C)
    ys = np.repeat(np.arange(C, dtype=float), T)

    # ------------------------ stable-layout figure ---------------------------
    plot_w = max(1.5, base_col_width * max(1, T))
    plot_h = max(1.5, base_row_height * max(1, C))

    fig_w = plot_w + gutter_width + 1.0   # extra buffer for legends
    fig_h = plot_h

    fig = plt.figure(
        figsize=(figsize if figsize is not None else (fig_w, fig_h)),
        constrained_layout=False,
        facecolor=background,
    )

    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[plot_w, gutter_width],  # dot grid area fixed
        wspace=0.02,
    )
    ax = fig.add_subplot(gs[0, 0], facecolor=background)

    # right gutter host (absorbs legends/labels, never alters grid)
    gutter_ax = fig.add_subplot(gs[0, 1], facecolor=background)
    gutter_ax.set_axis_off()
    gutter_ax.set_in_layout(False)

    # plot dots
    ax.scatter(xs, ys, s=sizes, c=colors, edgecolor=edgecolor, linewidth=edge_lw)

    # axes limits & ticks
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(C - 0.5, -0.5)
    ax.set_xticks(range(T))
    ax.set_xticklabels(time_order, rotation=xlabel_rotation, ha="center", fontsize=9)
    ax.set_yticks(range(C))
    ax.set_yticklabels(cluster_order, fontsize=9)

    # label styling
    _bold_first_rows(ax, row_dividers)
    _color_row_labels(ax, row_color_groups)

    # optional faint grid
    _draw_tick_grids(
        ax,
        n_x=T,
        n_y=C,
        tick_grid_x=tick_grid_x,
        tick_grid_y=tick_grid_y,
        tick_grid_lw=tick_grid_lw,
        tick_grid_alpha=tick_grid_alpha,
    )

    # horizontal row dividers
    _draw_row_dividers(
        ax,
        C=C,
        row_dividers=row_dividers,
        row_divider_color=row_divider_color,
        row_divider_lw=row_divider_lw,
        row_divider_alpha=row_divider_alpha,
    )

    _style_axes_spines(ax)
    ax.set_title(color if title is None else title, fontsize=12)

    # ------------------------ colorbar (figure-anchored) ---------------------
    if add_colorbar:
        left = 1.0 - gutter_pad - 0.12
        bottom = 0.7
        width = 0.05
        height = 0.15
        _add_vertical_colorbar(
            fig,
            norm,
            cmap_obj,
            left=left,
            bottom=bottom,
            width=width,
            height=height,
            title=cbar_title,
        )

    # ------------------------ size legend (figure-anchored) ------------------
    if show_size_legend:
        _add_fraction_size_legend(
            fig,
            s_min=s_min,
            s_max=s_max,
            size_legend_values=size_legend_values,
            size_legend_title=size_legend_title,
            size_legend_facecolor=size_legend_facecolor,
            left=1.0 - gutter_pad - 0.18,
            bottom=0.55,
            width=0.22,
            height=0.10,
            background=background,
        )

    # ------------------------ row color legend (optional) --------------------
    if show_row_color_legend and row_color_groups:
        _add_row_color_legend(
            fig,
            cluster_order,
            row_color_groups,
            left=1.0 - gutter_pad - 0.26,
            bottom=0.08,
            width=0.22,
            height=0.18,
            background=background,
        )

    # ------------------------ long table output ------------------------------
    if return_long:
        clust_rep = np.repeat(np.array(cluster_order), T)
        time_rep  = np.tile(np.array(time_order), C)
        long_out = pd.DataFrame({
            "cluster": clust_rep,
            "time": time_rep,
            "n": n.astype(np.int32),
            "k": k.astype(np.int32),
            "N_clust": np.repeat(N_clust, T).astype(np.int32),
            "N_time":  np.tile(N_time, C).astype(np.int32),
            "mean_expr": mean_expr.astype(float),
            "mean_expr_plot": vals_color.astype(float),
            "frac_expressing": np.where(n > 0, k / np.maximum(n, 1), np.nan),
            "size_signal": np.clip(
                np.where(n > 0, k / np.maximum(n, 1), np.nan), 0.0, 1.0
            ),
            "size_pixels": sizes.astype(float),
            "is_low_support": (~eligible).astype(bool),
        })
    else:
        long_out = None

    if not show:
        plt.close(ax.figure)

    return ax, long_out


# ---------------------------------------------------------------------
# 2) Dotplot: color feature vs studies
# ---------------------------------------------------------------------

def gene_groups_vs_studies(
    adata,
    color: str,
    *,
    groupby: str = "ZMAP_CellType",       # cluster column in .obs
    study_col: str = "study_id",          # study column in .obs (x-axis)
    layer: str | None = "tpm_log",
    use_raw: bool | None = None,
    detect_threshold: float = 0.0,        # > threshold => “expressing”
    show: bool = True,
    # ===== color (mean expression) =====
    cmap: str = "viridis",
    vmin: float | None = 0,
    vmax: float | None = None,
    standard_scale: str | None = None,    # None | "study" | "cluster"
    # ===== DOT SIZE (fraction expressing) =====
    s_min: float = 4.0,
    s_max: float = 60.0,
    size_zero_for_missing: bool = True,
    # ===== LOW-SUPPORT / MISSING =====
    abs_min_cells: int = 10,
    rel_min_frac: float = 0.01,
    rel_abs_cap: int = 300,
    low_support_grey: str = "0.7",
    low_support_alpha: float = 0.5,
    # ===== layout/labels =====
    base_col_width: float = 0.14,         # inches per study column (plot area)
    base_row_height: float = 0.14,        # inches per cluster row (plot area)
    gutter_width: float = 1.5,            # inches, fixed-width right gutter
    gutter_pad: float = 0.03,             # extra right-side pad (in figure coords)
    figsize: tuple | None = None,         # if None, computed from grid + gutter
    xlabel_rotation: int = 90,
    title: str | None = None,             # defaults to color name only
    add_colorbar: bool = True,
    cbar_title: str = "log(tpm)\ncounts",
    # ===== size legend =====
    show_size_legend: bool = True,
    size_legend_values: tuple | None = None,  # fractions to show (0–1)
    size_legend_title: str | None = None,
    size_legend_facecolor: str = "black",
    # ===== ordering & filtering =====
    cluster_order: list[str] | None = DEFAULT_CLUSTER_ORDER_ZMAP_CellType,
    study_order: list[str] | None = DEFAULT_STUDY_ORDER,
    omit_groups: list[str] | None = ['nan', 'unknown'],     # omits clusters
    # ===== cosmetics =====
    edgecolor: str = "black",
    edge_lw: float = 0.1,
    background: str = "white",
    # ===== faint tick/grid lines =====
    tick_grid_x: bool = False,
    tick_grid_y: bool = False,
    tick_grid_alpha: float = 0.15,
    tick_grid_lw: float = 0.6,
    # ===== Row color group mapping & legend =====
    row_color_groups: dict[str, list[str]] | None = DEFAULT_ROW_COLOR_GROUPS_ZMAP_CellType,
    show_row_color_legend: bool = False,
    # ===== Horizontal split lines between rows =====
    row_dividers: list[int] | None = DEFAULT_ROW_DIVIDERS_ZMAP_CellType,
    row_divider_color: str = "black",
    row_divider_lw: float = 0.5,
    row_divider_alpha: float = 1,
    # ===== performance/output controls =====
    return_long: bool = False,
):
    """
    Stable-layout dotplot of a single color feature across (cluster × study).

    `color` is usually a gene name (taken from raw/layer/X), but if no gene
    match is found it will fall back to a numeric column in `adata.obs`.

    Dot semantics
    -------------
    * Color: mean value per (cluster, study) bin.
    * Size:  fraction of cells with value > `detect_threshold`.
    * Low-support / missing: grey at minimum size.
    """
    color = str(color)

    # ------------------------ color vector ------------------------
    x, color_source = _extract_color_vector(adata, color, layer, use_raw)

    # ------------------------ metadata as arrays (no full copy) --------------
    obs = adata.obs
    if groupby not in obs or study_col not in obs:
        raise ValueError(f"Missing columns in adata.obs: need '{groupby}' and '{study_col}'.")

    clusters_ser = obs[groupby].astype(str)
    studies_ser  = obs[study_col].astype(str)

    # Optional omit
    if omit_groups:
        omit_set = set(map(str, omit_groups))
        keep_mask = ~clusters_ser.isin(omit_set)
        clusters_ser = clusters_ser[keep_mask]
        studies_ser  = studies_ser[keep_mask]
        x            = x[keep_mask.values]
    else:
        omit_set = set()

    # Determine orders (respect categorical if present)
    if cluster_order is None:
        if isinstance(obs[groupby].dtype, pd.CategoricalDtype):
            cluster_order = [
                c for c in obs[groupby].cat.categories.astype(str) if c not in omit_set
            ]
        else:
            cluster_order = list(pd.unique(clusters_ser))
    else:
        present = set(pd.unique(clusters_ser))
        cluster_order = [c for c in cluster_order if c in present]

    if study_order is None:
        if isinstance(obs[study_col].dtype, pd.CategoricalDtype):
            study_order = list(obs[study_col].cat.categories.astype(str))
        else:
            study_order = list(pd.unique(studies_ser))

    C = len(cluster_order)
    S = len(study_order)

    # Map to categorical codes aligned to the chosen orders
    clust_cat = pd.Categorical(clusters_ser, categories=cluster_order, ordered=True)
    study_cat = pd.Categorical(studies_ser,  categories=study_order,   ordered=True)
    c_codes = clust_cat.codes.astype(np.int64)
    s_codes = study_cat.codes.astype(np.int64)

    valid = (c_codes >= 0) & (s_codes >= 0)
    c_codes = c_codes[valid]
    s_codes = s_codes[valid]
    x       = x[valid]

    # Linearize (cluster,study) → idx
    lin = (c_codes * S + s_codes).astype(np.int64)

    # ------------------------ fast aggregation via bincount ------------------
    n = np.bincount(lin, minlength=C*S).astype(np.int32)
    is_expr = (x > detect_threshold)
    k = np.bincount(lin, weights=is_expr.astype(np.int8), minlength=C*S).astype(np.int32)
    sum_expr = np.bincount(lin, weights=x, minlength=C*S)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_expr = np.where(n > 0, sum_expr / n, np.nan)

    # N_clust per cluster, N_study per study
    N_clust = np.bincount(c_codes, minlength=C).astype(np.int32)
    N_study = np.bincount(s_codes, minlength=S).astype(np.int32)

    # ------------------------ low-support eligibility ------------------------
    req = np.maximum(
        abs_min_cells,
        np.minimum(np.ceil(N_clust * float(rel_min_frac)).astype(int), int(rel_abs_cap)),
    ).astype(np.int32)
    req_grid = np.repeat(req[:, None], S, axis=1).ravel()

    eligible = (n >= req_grid)
    has_cells = (n > 0)

    # ------------------------ standard scaling (optional) --------------------
    vals_color = mean_expr.copy()

    if standard_scale == "study":
        vals_color = vals_color.reshape(C, S)
        for ci in range(C):
            vals_color[ci, :] = _scale01(vals_color[ci, :])
        vals_color = vals_color.ravel()
    elif standard_scale == "cluster":
        vals_color = vals_color.reshape(C, S)
        for si in range(S):
            vals_color[:, si] = _scale01(vals_color[:, si])
        vals_color = vals_color.ravel()
    elif standard_scale is not None:
        raise ValueError("standard_scale must be None, 'study', or 'cluster'.")

    # ------------------------ color mapping ------------------------
    cmap_obj = plt.get_cmap(cmap)
    norm, _vmin, _vmax = _compute_color_norm(vals_color, vmin, vmax)

    colors = np.zeros((C*S, 4), dtype=float)
    valid_color = has_cells
    colors[valid_color] = cmap_obj(
        norm(np.nan_to_num(vals_color[valid_color], nan=_vmin))
    )

    # ------------------------ dot sizes (fraction expressing) ----------------
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(n > 0, k / np.maximum(n, 1), np.nan)

    sizes = _compute_sizes_from_fraction(
        frac, s_min=s_min, s_max=s_max, size_zero_for_missing=size_zero_for_missing
    )

    # Treat low-support and missing identically
    low_or_missing = (~eligible) | (~has_cells)
    if np.any(low_or_missing):
        colors[low_or_missing] = mpl.colors.to_rgba(low_support_grey, alpha=low_support_alpha)
        sizes[low_or_missing] = s_min

    # ------------------------ positions ---------------------------
    xs = np.tile(np.arange(S, dtype=float), C)
    ys = np.repeat(np.arange(C, dtype=float), S)

    # ------------------------ STABLE LAYOUT FIGURE ---------------------------
    plot_w = max(1.5, base_col_width * max(1, S))
    plot_h = max(1.5, base_row_height * max(1, C))
    fig_w = plot_w + gutter_width + 1.0
    fig_h = plot_h

    fig = plt.figure(
        figsize=(figsize if figsize is not None else (fig_w, fig_h)),
        constrained_layout=False,
        facecolor=background,
    )
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[plot_w, gutter_width],
        wspace=0.02,
    )

    ax = fig.add_subplot(gs[0, 0], facecolor=background)

    # right gutter host (empty)
    gutter_ax = fig.add_subplot(gs[0, 1], facecolor=background)
    gutter_ax.set_axis_off()
    gutter_ax.set_in_layout(False)

    # plot dots
    ax.scatter(xs, ys, s=sizes, c=colors, edgecolor=edgecolor, linewidth=edge_lw)

    # Exact limits so ticks align with dot centers
    ax.set_xlim(-0.5, S - 0.5)
    ax.set_ylim(C - 0.5, -0.5)

    # ticks & labels
    ax.set_xticks(range(S))
    ax.set_xticklabels(study_order, rotation=xlabel_rotation, ha="center", fontsize=9)
    ax.set_yticks(range(C))
    ax.set_yticklabels(cluster_order, fontsize=9)

    # label styling
    _bold_first_rows(ax, row_dividers)
    _color_row_labels(ax, row_color_groups)

    # Optional faint tick/grid
    _draw_tick_grids(
        ax,
        n_x=S,
        n_y=C,
        tick_grid_x=tick_grid_x,
        tick_grid_y=tick_grid_y,
        tick_grid_lw=tick_grid_lw,
        tick_grid_alpha=tick_grid_alpha,
    )

    # Horizontal row dividers
    _draw_row_dividers(
        ax,
        C=C,
        row_dividers=row_dividers,
        row_divider_color=row_divider_color,
        row_divider_lw=row_divider_lw,
        row_divider_alpha=row_divider_alpha,
    )

    _style_axes_spines(ax)
    ax.set_title(color if title is None else title, fontsize=12)

    # ------------------------ colorbar (figure-anchored) ---------------------
    if add_colorbar:
        left = 1.0 - gutter_pad - 0.12
        bottom = 0.70
        width = 0.05
        height = 0.15
        _add_vertical_colorbar(
            fig,
            norm,
            cmap_obj,
            left=left,
            bottom=bottom,
            width=width,
            height=height,
            title=cbar_title,
        )

    # ------------------------ size legend (figure-anchored) ------------------
    if show_size_legend:
        _add_fraction_size_legend(
            fig,
            s_min=s_min,
            s_max=s_max,
            size_legend_values=size_legend_values,
            size_legend_title=size_legend_title,
            size_legend_facecolor=size_legend_facecolor,
            left=1.0 - gutter_pad - 0.18,
            bottom=0.55,
            width=0.22,
            height=0.10,
            background=background,
        )

    # ------------------------ row color legend (optional) --------------------
    if show_row_color_legend and row_color_groups:
        _add_row_color_legend(
            fig,
            cluster_order,
            row_color_groups,
            left=1.0 - gutter_pad - 0.26,
            bottom=0.08,
            width=0.22,
            height=0.18,
            background=background,
        )

    # ------------------------ long table output ------------------------------
    if return_long:
        clust_rep = np.repeat(np.array(cluster_order), S)
        study_rep = np.tile(np.array(study_order), C)
        long_out = pd.DataFrame({
            "cluster": clust_rep,
            "study": study_rep,
            "n": n.astype(np.int32),
            "k": k.astype(np.int32),
            "N_clust": np.repeat(N_clust, S).astype(np.int32),
            "N_study": np.tile(N_study, C).astype(np.int32),
            "mean_expr": mean_expr.astype(float),
            "mean_expr_plot": vals_color.astype(float),
            "frac_expressing": np.where(n > 0, k / np.maximum(n, 1), np.nan),
            "size_signal": np.clip(
                np.where(n > 0, k / np.maximum(n, 1), np.nan), 0.0, 1.0
            ),
            "size_pixels": sizes.astype(float),
            "is_low_support": (~eligible).astype(bool),
        })
    else:
        long_out = None

    if not show:
        plt.close(ax.figure)

    return ax, long_out


# ---------------------------------------------------------------------
# 3) Combined dotplot: time and studies in one figure
# ---------------------------------------------------------------------

def gene_groups_vs_time_and_studies(
    adata,
    color: str,
    *,
    # shared metadata
    groupby: str = "ZMAP_CellType",
    time_col: str = "time_group_id",
    study_col: str = "study_id",
    layer: str | None = "tpm_log",
    use_raw: bool | None = None,
    detect_threshold: float = 0.0,
    show: bool = True,
    # ===== COLOR (mean expression) shared across both panels =====
    cmap: str = "viridis",
    vmin: float | None = 0,
    vmax: float | None = None,
    cbar_title: str = "log(tpm)\ncounts",
    add_colorbar: bool = True,
    # ===== DOT SIZE (fraction expressing) shared mapping =====
    s_min: float = 4.0,
    s_max: float = 60.0,
    size_zero_for_missing: bool = True,
    # ===== LOW-SUPPORT / MISSING =====
    abs_min_cells: int = 10,
    rel_min_frac: float = 0.01,
    rel_abs_cap: int = 300,
    low_support_grey: str = "0.7",
    low_support_alpha: float = 0.5,
    # ===== layout/labels (SQUARE cells) =====
    cell_size: float = 0.14,     # inches per data cell in BOTH x and y (for both panels)
    middle_gap: float = 0.2,    # inches between panels (blank spacer)
    gutter_width: float = 0.8,   # inches for right gutter (legends live here)
    gutter_pad: float = 0.03,    # right-side pad in figure coords (for cbar/legends)
    figsize: tuple | None = None,
    xlabel_rotation: int = 90,
    title_left: str | None = "Timepoints",
    title_right: str | None = "Studies",
    # ===== size legend (shared) =====
    show_size_legend: bool = True,
    size_legend_values: tuple | None = None,
    size_legend_title: str | None = None,
    size_legend_facecolor: str = "black",
    # ===== ordering & filtering =====
    cluster_order: list[str] | None = DEFAULT_CLUSTER_ORDER_ZMAP_CellType,
    time_order: list[str] | None = None,
    study_order: list[str] | None = DEFAULT_STUDY_ORDER,
    omit_groups: list[str] | None = ['nan', 'unknown'],
    # ===== cosmetics =====
    edgecolor: str = "black",
    edge_lw: float = 0.1,
    background: str = "white",
    # ===== faint tick/grid lines =====
    tick_grid_x_left: bool = False,
    tick_grid_x_right: bool = False,
    tick_grid_y: bool = False,
    tick_grid_alpha: float = 0.15,
    tick_grid_lw: float = 0.6,
    # ===== Row color group mapping & legend =====
    row_color_groups: dict[str, list[str]] | None = DEFAULT_ROW_COLOR_GROUPS_ZMAP_CellType,
    show_row_color_legend: bool = False,
    # ===== Horizontal split lines between rows =====
    row_dividers: list[int] | None = DEFAULT_ROW_DIVIDERS_ZMAP_CellType,
    row_divider_color: str = "black",
    row_divider_lw: float = 0.5,
    row_divider_alpha: float = 1.0,
    # ===== output controls =====
    return_long: bool = False,
):
    """
    Two-panel dotplot of a single color feature across both (cluster × time)
    and (cluster × study), sharing the same color scale and size mapping.

    `color` is usually a gene name (taken from raw/layer/X), but if no gene
    match is found it will fall back to a numeric column in `adata.obs`.

    Left panel:
        rows = clusters, columns = timepoints.
    Right panel:
        rows = clusters, columns = studies.

    Dot semantics (both panels)
    ---------------------------
    * Color: mean value per bin.
    * Size:  fraction of cells with value > `detect_threshold`.
    * Low-support / missing: grey at minimum size.
    """
    color = str(color)

    # ------------------------ color vector ------------------------
    x_full, color_source = _extract_color_vector(adata, color, layer, use_raw)

    obs = adata.obs
    if groupby not in obs or time_col not in obs or study_col not in obs:
        raise ValueError(
            f"Missing columns in adata.obs: need '{groupby}', '{time_col}', and '{study_col}'."
        )

    clusters_ser = obs[groupby].astype(str)
    times_ser    = obs[time_col].astype(str)
    studies_ser  = obs[study_col].astype(str)

    # Optional omit
    if omit_groups:
        omit_set = set(map(str, omit_groups))
        keep_mask = ~clusters_ser.isin(omit_set)
        clusters_ser = clusters_ser[keep_mask]
        times_ser    = times_ser[keep_mask]
        studies_ser  = studies_ser[keep_mask]
        x_full       = x_full[keep_mask.values]
    else:
        omit_set = set()

    # Row order (clusters) once
    if cluster_order is None:
        if isinstance(obs[groupby].dtype, pd.CategoricalDtype):
            cluster_order = [
                c for c in obs[groupby].cat.categories.astype(str) if c not in omit_set
            ]
        else:
            cluster_order = list(pd.unique(clusters_ser))
    else:
        present = set(pd.unique(clusters_ser))
        cluster_order = [c for c in cluster_order if c in present]
    C = len(cluster_order)

    # Column orders
    if time_order is None:
        if isinstance(obs[time_col].dtype, pd.CategoricalDtype):
            time_order = list(obs[time_col].cat.categories.astype(str))
        else:
            time_order = list(pd.unique(times_ser))
    if study_order is None:
        if isinstance(obs[study_col].dtype, pd.CategoricalDtype):
            study_order = list(obs[study_col].cat.categories.astype(str))
        else:
            study_order = list(pd.unique(studies_ser))
    T, S = len(time_order), len(study_order)

    # Codes
    clust_cat_t = pd.Categorical(clusters_ser, categories=cluster_order, ordered=True)
    time_cat    = pd.Categorical(times_ser,    categories=time_order,  ordered=True)
    clust_cat_s = pd.Categorical(clusters_ser, categories=cluster_order, ordered=True)
    study_cat   = pd.Categorical(studies_ser,  categories=study_order,   ordered=True)

    c_codes_t = clust_cat_t.codes.astype(np.int64)
    t_codes   = time_cat.codes.astype(np.int64)
    c_codes_s = clust_cat_s.codes.astype(np.int64)
    s_codes   = study_cat.codes.astype(np.int64)

    valid_t = (c_codes_t >= 0) & (t_codes >= 0)
    valid_s = (c_codes_s >= 0) & (s_codes >= 0)

    # -------- aggregation helper (local on purpose) --------
    def _aggregate(c_codes, x_codes, Xvec, C, Xn):
        lin = (c_codes * Xn + x_codes).astype(np.int64)
        n = np.bincount(lin, minlength=C*Xn).astype(np.int32)
        is_expr = (Xvec > detect_threshold)
        k = np.bincount(lin, weights=is_expr.astype(np.int8), minlength=C*Xn).astype(np.int32)
        sum_expr = np.bincount(lin, weights=Xvec, minlength=C*Xn)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_expr = np.where(n > 0, sum_expr / n, np.nan)

        N_clust = np.bincount(c_codes, minlength=C).astype(np.int32)
        N_x     = np.bincount(x_codes, minlength=Xn).astype(np.int32)

        req = np.maximum(
            abs_min_cells,
            np.minimum(np.ceil(N_clust * float(rel_min_frac)).astype(int), int(rel_abs_cap)),
        ).astype(np.int32)
        req_grid = np.repeat(req[:, None], Xn, axis=1).ravel()

        eligible = (n >= req_grid)
        has_cells = (n > 0)

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(n > 0, k / np.maximum(n, 1), np.nan)

        return n, k, mean_expr, frac, eligible, has_cells, N_clust, N_x

    n_t, k_t, mean_t, frac_t, elig_t, has_t, Nclust_t, Ntime = _aggregate(
        c_codes_t[valid_t], t_codes[valid_t], x_full[valid_t], C, T,
    )
    n_s, k_s, mean_s, frac_s, elig_s, has_s, Nclust_s, Nstudy = _aggregate(
        c_codes_s[valid_s], s_codes[valid_s], x_full[valid_s], C, S,
    )

    # ----- absolute color values (no standardization) -----
    vals_t = mean_t.copy()
    vals_s = mean_s.copy()

    # Shared color scale across both panels
    cmap_obj = plt.get_cmap(cmap)
    finite_all = np.concatenate([
        vals_t[np.isfinite(vals_t)],
        vals_s[np.isfinite(vals_s)],
    ])
    if vmin is not None:
        _vmin = vmin
    else:
        _vmin = np.nanpercentile(finite_all, 1) if finite_all.size else 0.0
    if vmax is not None:
        _vmax = vmax
    else:
        _vmax = np.nanpercentile(finite_all, 99) if finite_all.size else 1.0
    norm = mpl.colors.Normalize(vmin=_vmin, vmax=_vmax)

    colors_t = np.zeros((C*T, 4), dtype=float)
    colors_s = np.zeros((C*S, 4), dtype=float)
    if np.any(has_t):
        colors_t[has_t] = cmap_obj(
            norm(np.nan_to_num(vals_t[has_t], nan=_vmin))
        )
    if np.any(has_s):
        colors_s[has_s] = cmap_obj(
            norm(np.nan_to_num(vals_s[has_s], nan=_vmin))
        )

    # Sizes (fraction expressing) – shared s_min, s_max, size_zero_for_missing
    sizes_t = _compute_sizes_from_fraction(
        frac_t, s_min=s_min, s_max=s_max, size_zero_for_missing=size_zero_for_missing
    )
    sizes_s = _compute_sizes_from_fraction(
        frac_s, s_min=s_min, s_max=s_max, size_zero_for_missing=size_zero_for_missing
    )

    # Low-support / missing
    low_or_missing_t = (~elig_t) | (~has_t)
    low_or_missing_s = (~elig_s) | (~has_s)
    if np.any(low_or_missing_t):
        colors_t[low_or_missing_t] = mpl.colors.to_rgba(
            low_support_grey, alpha=low_support_alpha
        )
        sizes_t[low_or_missing_t] = s_min
    if np.any(low_or_missing_s):
        colors_s[low_or_missing_s] = mpl.colors.to_rgba(
            low_support_grey, alpha=low_support_alpha
        )
        sizes_s[low_or_missing_s] = s_min

    # Positions
    xs_t = np.tile(np.arange(T, dtype=float), C)
    ys_t = np.repeat(np.arange(C, dtype=float), T)
    xs_s = np.tile(np.arange(S, dtype=float), C)
    ys_s = np.repeat(np.arange(C, dtype=float), S)

    # ===================== LAYOUT IN CELL UNITS =====================
    gap_cells    = max(0.0, middle_gap / max(cell_size, 1e-9))
    gutter_cells = max(0.0, gutter_width / max(cell_size, 1e-9))

    fig_w_cells = T + gap_cells + S + gutter_cells
    fig_h_cells = C
    if figsize is None:
        fig_w = fig_w_cells * cell_size
        fig_h = fig_h_cells * cell_size
    else:
        fig_w, fig_h = figsize

    fig = plt.figure(
        figsize=(fig_w, fig_h),
        constrained_layout=False,
        facecolor=background,
    )

    gs = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=[T, gap_cells, S, gutter_cells],
        wspace=0.0,
    )
    ax_left  = fig.add_subplot(gs[0, 0], facecolor=background)
    spacer   = fig.add_subplot(gs[0, 1], facecolor=background)
    spacer.set_axis_off()
    ax_right = fig.add_subplot(gs[0, 2], facecolor=background)
    gutter_ax = fig.add_subplot(gs[0, 3], facecolor=background)
    gutter_ax.set_axis_off()
    gutter_ax.set_in_layout(False)
    # ===============================================================

    # Draw dots
    ax_left.scatter(
        xs_t, ys_t, s=sizes_t, c=colors_t, marker="o",
        edgecolor=edgecolor, linewidth=edge_lw,
    )
    ax_right.scatter(
        xs_s, ys_s, s=sizes_s, c=colors_s, marker="o",
        edgecolor=edgecolor, linewidth=edge_lw,
    )

    # Limits & ticks
    ax_left.set_xlim(-0.5, T - 0.5)
    ax_left.set_ylim(C - 0.5, -0.5)
    ax_right.set_xlim(-0.5, S - 0.5)
    ax_right.set_ylim(C - 0.5, -0.5)

    ax_left.set_xticks(range(T))
    ax_left.set_xticklabels(time_order, rotation=xlabel_rotation, ha="center", fontsize=9)
    ax_right.set_xticks(range(S))
    ax_right.set_xticklabels(study_order, rotation=xlabel_rotation, ha="center", fontsize=9)

    ax_left.set_yticks(range(C))
    ax_left.set_yticklabels(cluster_order, fontsize=9)
    ax_right.set_yticks(range(C))
    ax_right.set_yticklabels([""] * C)

    # Label styling on left
    _bold_first_rows(ax_left, row_dividers)
    _color_row_labels(ax_left, row_color_groups)

    # Optional faint grid
    _draw_tick_grids(
        ax_left,
        n_x=T,
        n_y=C,
        tick_grid_x=tick_grid_x_left,
        tick_grid_y=tick_grid_y,
        tick_grid_lw=tick_grid_lw,
        tick_grid_alpha=tick_grid_alpha,
    )
    _draw_tick_grids(
        ax_right,
        n_x=S,
        n_y=C,
        tick_grid_x=tick_grid_x_right,
        tick_grid_y=tick_grid_y,
        tick_grid_lw=tick_grid_lw,
        tick_grid_alpha=tick_grid_alpha,
    )

    # Row dividers
    _draw_row_dividers(
        ax_left,
        C=C,
        row_dividers=row_dividers,
        row_divider_color=row_divider_color,
        row_divider_lw=row_divider_lw,
        row_divider_alpha=row_divider_alpha,
    )
    _draw_row_dividers(
        ax_right,
        C=C,
        row_dividers=row_dividers,
        row_divider_color=row_divider_color,
        row_divider_lw=row_divider_lw,
        row_divider_alpha=row_divider_alpha,
    )

    # cosmetics
    _style_axes_spines(ax_left)
    _style_axes_spines(ax_right)

    # Titles over each panel (keep small)
    if title_left:
        ax_left.set_title(f"{title_left}", fontsize=8)
    if title_right:
        ax_right.set_title(f"{title_right}", fontsize=8)

    # ------------------------ SHARED colorbar in gutter + rotated label -----
    if add_colorbar:
        left_cb = 1.0 - gutter_pad - 0.12
        bottom_cb = 0.70
        width_cb = 0.02
        height_cb = 0.15
        _add_vertical_colorbar(
            fig,
            norm,
            cmap_obj,
            left=left_cb,
            bottom=bottom_cb,
            width=width_cb,
            height=height_cb,
            title=cbar_title,
        )

        # Rotated label: centered vertically, to the LEFT of the colorbar
        gene_pad = 0.025  # horizontal spacing from cbar
        fig.text(
            left_cb - gene_pad,
            bottom_cb + height_cb / 2.0,
            color,
            rotation=90,
            rotation_mode="anchor",
            va="center",
            ha="center",
            fontsize=10,
        )

    # ------------------------ SHARED size legend in gutter -------------------
    if show_size_legend:
        _add_fraction_size_legend(
            fig,
            s_min=s_min,
            s_max=s_max,
            size_legend_values=size_legend_values,
            size_legend_title=size_legend_title,
            size_legend_facecolor=size_legend_facecolor,
            left=1.0 - gutter_pad - 0.14,
            bottom=0.55,
            width=0.12,
            height=0.10,
            background=background,
        )

    # ------------------------ optional row-color legend (gutter) -------------
    if show_row_color_legend and row_color_groups:
        _add_row_color_legend(
            fig,
            cluster_order,
            row_color_groups,
            left=1.0 - gutter_pad - 0.26,
            bottom=0.08,
            width=0.22,
            height=0.18,
            background=background,
        )

    # ------------------------ optional returns -------------------------------
    long_time = None
    long_study = None
    if return_long:
        clust_rep_t = np.repeat(np.array(cluster_order), T)
        time_rep    = np.tile(np.array(time_order), C)
        long_time = pd.DataFrame({
            "panel": "time",
            "cluster": clust_rep_t,
            "time": time_rep,
            "n": n_t.astype(np.int32),
            "k": k_t.astype(np.int32),
            "mean_expr": mean_t.astype(float),
            "mean_expr_plot": vals_t.astype(float),
            "frac_expressing": np.where(n_t > 0, k_t / np.maximum(n_t, 1), np.nan),
            "size_pixels": sizes_t.astype(float),
            "is_low_support": (~elig_t).astype(bool),
        })
        clust_rep_s = np.repeat(np.array(cluster_order), S)
        study_rep   = np.tile(np.array(study_order), C)
        long_study = pd.DataFrame({
            "panel": "study",
            "cluster": clust_rep_s,
            "study": study_rep,
            "n": n_s.astype(np.int32),
            "k": k_s.astype(np.int32),
            "mean_expr": mean_s.astype(float),
            "mean_expr_plot": vals_s.astype(float),
            "frac_expressing": np.where(n_s > 0, k_s / np.maximum(n_s, 1), np.nan),
            "size_pixels": sizes_s.astype(float),
            "is_low_support": (~elig_s).astype(bool),
        })

    if not show:
        plt.close(fig)

    return (ax_left, ax_right), (long_time, long_study)
