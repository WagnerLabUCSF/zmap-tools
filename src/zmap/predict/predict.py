# -------------------------------------------------------------------
#  ZMAP  —  Prediction, kNN transfer, preprocessing, diagnostics
# -------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Mapping

import warnings, os, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from scipy import sparse
from adjustText import adjust_text
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve)

from sklearn.preprocessing import label_binarize
from .knn_backend import knn_search

# ---- IPython display (optional) ----
try:
    from IPython.display import display as _ipython_display
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False


def _display_df(df: pd.DataFrame, max_rows: int | None = None) -> None:
    """Display a DataFrame via IPython if available, else print."""
    show_df = df.head(max_rows) if max_rows is not None else df
    if _HAS_IPYTHON:
        _ipython_display(show_df)
    else:
        print(show_df.to_string(index=False))
    if max_rows is not None and len(df) > max_rows:
        print(f"       … {len(df) - max_rows:,} additional rows")


# ---- Timestamped logging ----
_T_PIPELINE_START: float | None = None


def _zlog(msg: str) -> None:
    """Print a ``[ZMAP HH:MM:SS]`` timestamped message."""
    if _T_PIPELINE_START is not None:
        elapsed = time.time() - _T_PIPELINE_START
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        ts = f"{h}:{m:02d}:{s:02d}"
    else:
        ts = "—"
    print(f"[ZMAP {ts}] {msg}")


# ---- Colormap routing: ref_label_col → reference uns key ----
_COLORMAP_UNS_KEY = {
    "ZMAP_CellType":  "ZMAP_colormap_C79",
    "ZMAP_Tissue":    "ZMAP_colormap_T28",
    "ZMAP_GermLayer": "ZMAP_colormap_G7",
}

# ---- Marker level routing: ref_label_col → load_consensus_markers level ----
_MARKER_LEVEL_KEY = {
    "ZMAP_CellType":  "CellType",
    "ZMAP_Tissue":    "Tissue",
    "ZMAP_GermLayer": "GermLayer",
}


# ================================================================
#  0. Preprocessing & Helpter Functions
# ================================================================

def preprocess_adata_query(
    adata_query: ad.AnnData,
    *,
    counts_source: str,              # explicit: "X" or layer name
    target_sum: float = 1e6,         # TPM-like library size
    inplace: bool = True,
    integer_tol: float = 1e-3,       # integer-like check tolerance
    strict_counts: bool = False,     # if True: error on non-count-ish data
) -> ad.AnnData:
    """
    Normalize raw counts in a query AnnData for ZMAP/Symphony label transfer.

    Reads raw counts from the specified location, performs library-size
    normalization (TPM-style) followed by log1p, and writes the result into
    ``adata.X``. Preprocessing metadata is recorded in
    ``adata.uns['ZMAP_preprocessing']['query']``.

    This function is called automatically by ``annotate_with_zmap`` when
    ``do_preprocess=True``. Call it manually only if you need fine-grained
    control over normalization before running the pipeline.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Query dataset. Modified in-place when ``inplace=True``.
    counts_source : str
        Where raw integer counts are stored. Pass ``"X"`` to use ``adata.X``,
        or a layer name (e.g. ``"counts"``) to use ``adata.layers[counts_source]``.
        This parameter is required and has no default — you must be explicit.
    target_sum : float, default ``1e6``
        Library size each cell is normalized to before log1p. The default
        produces TPM-scale values (counts per million).
    inplace : bool, default ``True``
        If ``True``, modify ``adata_query`` in-place and return it.
        If ``False``, operate on a copy and return the copy.
    integer_tol : float, default ``1e-3``
        Tolerance used when checking whether values are integer-like. Values
        deviating from the nearest integer by more than this amount count
        towards the non-integer fraction.
    strict_counts : bool, default ``False``
        If ``True``, raise a ``ValueError`` when the data contains NaN/inf,
        negative values, or appears non-integer-like (> 1% of non-zero values
        deviate from an integer). If ``False``, emit a warning instead.

    Returns
    -------
    anndata.AnnData
        The preprocessed AnnData (same object when ``inplace=True``).

    Raises
    ------
    KeyError
        If ``counts_source`` is not ``"X"`` and is not found in ``adata.layers``.
    TypeError
        If the raw data is not numeric.
    ValueError
        If ``strict_counts=True`` and data quality checks fail.

    Notes
    -----
    After this call, ``adata.X`` contains log-normalized (TPM + log1p) values
    regardless of what was in ``adata.X`` before. The original counts in
    ``counts_source`` are not modified.
    """
    if not inplace:
        adata = adata_query.copy()
    else:
        adata = adata_query

    # ---- 1. Raw counts ----
    if counts_source == "X":
        X_raw = adata.X
        source_descr = ".X"
    else:
        if counts_source not in adata.layers:
            raise KeyError(
                f"counts_source='{counts_source}' not found in adata.layers. "
                "Use 'X' or a valid layer name."
            )
        X_raw = adata.layers[counts_source]
        source_descr = f"layers['{counts_source}']"

    if sparse.issparse(X_raw):
        data = X_raw.data
    else:
        X_raw = np.asarray(X_raw)
        data = X_raw.ravel()

    # ---- 2. Sanity checks ----
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"Raw data in {source_descr} are not numeric.")

    finite_mask = np.isfinite(data)
    if not finite_mask.all():
        msg = "Raw counts contain NaN/inf values."
        if strict_counts:
            raise ValueError(msg)
        warnings.warn(msg)

    data_finite = data[finite_mask]
    if np.any(data_finite < 0):
        msg = "Raw counts contain negative values."
        if strict_counts:
            raise ValueError(msg)
        warnings.warn(msg)

    # integer-like check (for counts)
    nonzero = data_finite[data_finite > 0]
    if nonzero.size > 0:
        sample = nonzero if nonzero.size <= 1_000_000 else np.random.default_rng(0).choice(nonzero, 1_000_000, replace=False)
        frac = np.abs(sample - np.round(sample))
        if np.mean(frac > integer_tol) > 0.01:
            msg = (
                f"Raw data in {source_descr} do not appear integer-like "
                f"({np.mean(frac>integer_tol)*100:.1f}% deviate > {integer_tol})."
            )
            if strict_counts:
                raise ValueError(msg)
            warnings.warn(msg)

    # ---- 3. Library-size normalization (TPM-ish) ----
    if sparse.issparse(X_raw):
        X_counts = X_raw.tocsr(copy=True)
        libsize = np.array(X_counts.sum(axis=1)).ravel()
        scale = np.ones_like(libsize)
        nz = libsize > 0
        scale[nz] = target_sum / libsize[nz]
        X_tpm = sparse.diags(scale) @ X_counts
    else:
        X_counts = np.array(X_raw, float, copy=True)
        libsize = X_counts.sum(axis=1)
        scale = np.ones_like(libsize)
        nz = libsize > 0
        scale[nz] = target_sum / libsize[nz]
        X_tpm = X_counts * scale[:, None]

    # ---- 4. log1p ----
    if sparse.issparse(X_tpm):
        X_tpm = X_tpm.tocsr()
        X_tpm.data = np.log1p(X_tpm.data)
        adata.X = X_tpm
    else:
        adata.X = np.log1p(X_tpm)

    # ---- 5. bookkeeping ----
    adata.uns.setdefault("ZMAP_preprocessing", {})
    adata.uns["ZMAP_preprocessing"]["query"] = {
        "counts_source": counts_source,
        "effective_source": source_descr,
        "target_sum": float(target_sum),
        "integer_tol": float(integer_tol),
        "strict_counts": bool(strict_counts),
    }
    return adata


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize (float32, contiguous) for kNN metric stability."""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _has_valid_tissue_labels(adata_query, tissue_col: str) -> bool:
    """Return True when query has at least one non-empty tissue label."""
    if tissue_col not in adata_query.obs:
        return False
    s = adata_query.obs[tissue_col]
    if s.isna().all():
        return False
    s_norm = pd.Series(s, copy=False).astype("string").str.strip().str.lower()
    valid = ~(s_norm.isna() | s_norm.isin({"", "nan", "none", "na"}))
    return bool(valid.any())


def _predict_pseudo_tissue_knn(
    adata_query,
    *,
    X_ref: np.ndarray,
    X_query: np.ndarray,
    ref_tissue: np.ndarray,
    query_tissue_col: str,
    ref_tissue_col: str,
    n_neighbors: int,
    metric: str,
    knn_backend: str,
    knn_device: str,
    knn_nprobe: int | None,
    pseudo_tissue_k: int | None,
    pseudo_tissue_threshold: float = 0.0,
    pseudo_tissue_margin_threshold: float = 0.0,
    unknown_label: str = "unknown",
    pseudo_col: str | None = None,
    faiss_cache_prefix: str | None = None,
    write_query_tissue_col: bool = True,
    plot_qc: bool = False,
    save_qc: bool = True,
    output_dir: str = "zmap_predict",
) -> dict[str, Any]:
    """
    Infer query tissue labels from reference tissues via plain kNN voting.

    Writes pseudo diagnostics to ``adata_query.obs`` and writes assigned
    tissues to ``adata_query.obs[query_tissue_col]``.
    """
    if X_ref.shape[0] <= 0:
        raise ValueError("No reference rows available for pseudo tissue prediction.")

    if pseudo_tissue_k is None:
        k_use = min(int(X_ref.shape[0]), max(int(n_neighbors), 31))
    else:
        k_use = min(int(X_ref.shape[0]), max(1, int(pseudo_tissue_k)))

    idx, dist, knn_meta = knn_search(
        X_ref,
        X_query,
        n_neighbors=int(k_use),
        metric=str(metric),
        backend=str(knn_backend),
        device=str(knn_device),
        nprobe=(None if knn_nprobe is None else int(knn_nprobe)),
        cache_key=(
            None
            if faiss_cache_prefix is None
            else f"{str(faiss_cache_prefix)}|pseudo_tissue"
        ),
    )

    ref_tissue_arr = np.asarray(ref_tissue, dtype=object)
    nbr_labels = ref_tissue_arr[idx]  # (n_query, k_use)

    n_q = int(nbr_labels.shape[0])
    pred = np.full(n_q, str(unknown_label), dtype=object)
    max_prob = np.zeros(n_q, dtype=float)
    margin = np.zeros(n_q, dtype=float)
    entropy = np.zeros(n_q, dtype=float)

    for i in range(n_q):
        vals = pd.Series(nbr_labels[i], copy=False).dropna().astype(str).to_numpy()
        if vals.size == 0:
            continue
        uniq, cnt = np.unique(vals, return_counts=True)
        order = np.argsort(cnt)[::-1]
        probs = cnt.astype(float) / float(cnt.sum())
        top_prob = float(probs[order[0]])
        second_prob = float(probs[order[1]]) if probs.size > 1 else 0.0
        top_label = str(uniq[order[0]])
        max_prob[i] = top_prob
        margin[i] = top_prob - second_prob
        entropy[i] = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        pred[i] = top_label

    pseudo_col_use = str(pseudo_col) if pseudo_col is not None else f"{query_tissue_col}_pseudo"
    adata_query.obs[pseudo_col_use] = pd.Series(pred, index=adata_query.obs.index, dtype="string")
    adata_query.obs[f"{pseudo_col_use}_max_prob"] = max_prob
    adata_query.obs[f"{pseudo_col_use}_margin"] = margin
    adata_query.obs[f"{pseudo_col_use}_entropy"] = entropy
    # Compatibility aliases used by plotting helpers in older API.
    adata_query.obs[f"{pseudo_col_use}_knn_max_prob"] = max_prob
    adata_query.obs[f"{pseudo_col_use}_knn_margin"] = margin
    adata_query.obs[f"{pseudo_col_use}_knn_entropy"] = entropy

    thr = float(pseudo_tissue_threshold)
    mar_thr = float(pseudo_tissue_margin_threshold)
    keep_hi = np.ones(n_q, dtype=bool)
    if thr > 0:
        keep_hi &= max_prob >= thr
    if mar_thr > 0:
        keep_hi &= margin >= mar_thr
    filtered = np.full(n_q, str(unknown_label), dtype=object)
    filtered[keep_hi] = pred[keep_hi]

    if bool(write_query_tissue_col):
        adata_query.obs[query_tissue_col] = pd.Series(
            filtered, index=adata_query.obs.index, dtype="string"
        )
    n_assigned_raw = int(np.sum(pd.Series(pred).astype("string") != str(unknown_label)))
    n_assigned_filtered = int(np.sum(pd.Series(filtered).astype("string") != str(unknown_label)))

    if bool(plot_qc):
        os.makedirs(output_dir, exist_ok=True)

        fig1 = plt.figure()
        plt.hist(pd.Series(max_prob).dropna(), bins=100, color="steelblue", alpha=0.7)
        if thr > 0:
            plt.axvline(thr, color="red", linestyle="--", label=f"threshold={thr:g}")
        plt.title(f"Tissue Predicted Probability\n{int(np.sum(keep_hi))} pass / {n_q} total")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Cell Count")
        if plt.gca().get_legend_handles_labels()[1]:
            plt.legend()
        plt.tight_layout()
        if bool(save_qc):
            prob_path = os.path.join(output_dir, f"{pseudo_col_use}_qc_probability.png")
            fig1.savefig(prob_path, dpi=300)
            _zlog(f"Saved tissue QC plot: {prob_path}")
        plt.show()

        fig2 = plt.figure()
        plt.hist(pd.Series(margin).dropna(), bins=100, color="steelblue", alpha=0.7)
        if mar_thr > 0:
            plt.axvline(
                mar_thr,
                color="red",
                linestyle="--",
                label=f"margin_threshold={mar_thr:g}",
            )
        plt.title(f"Tissue Predicted Margin\n{int(np.sum(keep_hi))} pass / {n_q} total")
        plt.xlabel("Predicted Margin")
        plt.ylabel("Cell Count")
        if plt.gca().get_legend_handles_labels()[1]:
            plt.legend()
        plt.tight_layout()
        if bool(save_qc):
            margin_path = os.path.join(output_dir, f"{pseudo_col_use}_qc_margin.png")
            fig2.savefig(margin_path, dpi=300)
            _zlog(f"Saved tissue QC plot: {margin_path}")
        plt.show()

    adata_query.uns.setdefault("zmap_pseudo_tissue", {})
    adata_query.uns["zmap_pseudo_tissue"] = {
        "enabled": True,
        "source": "predict_labels_tissue_kNN:auto",
        "ref_tissue_col": str(ref_tissue_col),
        "query_tissue_col": str(query_tissue_col),
        "pseudo_col": str(pseudo_col_use),
        "k": int(k_use),
        "threshold": float(thr),
        "margin_threshold": float(mar_thr),
        "unknown_label": str(unknown_label),
        "knn_backend_requested": knn_meta.get("backend_requested", knn_backend),
        "knn_device_requested": knn_meta.get("device_requested", knn_device),
        "knn_nprobe_requested": (None if knn_nprobe is None else int(knn_nprobe)),
        "knn_backend_used": knn_meta.get("backend_used", "sklearn"),
        "knn_device_used": knn_meta.get("device_used", "cpu"),
        "n_query": int(adata_query.n_obs),
        "n_assigned_non_unknown_raw": int(n_assigned_raw),
        "n_assigned_non_unknown": int(n_assigned_filtered),
        "n_pass_thresholds": int(np.sum(keep_hi)),
        "write_query_tissue_col": bool(write_query_tissue_col),
    }

    return {
        "k": int(k_use),
        "pseudo_col": str(pseudo_col_use),
        "n_assigned_non_unknown_raw": int(n_assigned_raw),
        "n_assigned_non_unknown": int(n_assigned_filtered),
        "n_pass_thresholds": int(np.sum(keep_hi)),
        "threshold": float(thr),
        "margin_threshold": float(mar_thr),
        "knn_meta": knn_meta,
    }


def _compute_tissue_aware_neighbors(
    *,
    X_ref: np.ndarray,
    X_query: np.ndarray,
    ref_tissue: np.ndarray | None,
    query_tissue: np.ndarray | None,
    n_neighbors: int,
    metric: str,
    tissue_mode: str,
    tissue_penalty_lambda: float,
    knn_backend: str,
    knn_device: str,
    knn_nprobe: int | None,
    hard_allow_global_fallback: bool = True,
    hard_fallback_min_cells: int | None = None,
    faiss_cache_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Compute neighbor graph with optional tissue-aware constraints.

    Returns
    -------
    (indices, distances, knn_meta)
        indices shape = (n_query, k), distances shape = (n_query, k)
    """
    mode = str(tissue_mode).lower()
    if mode not in {"none", "hard", "soft"}:
        raise ValueError("tissue_mode must be one of {'none', 'hard', 'soft'}.")

    k = int(n_neighbors)
    if k <= 0:
        raise ValueError("n_neighbors must be positive.")
    if X_ref.shape[0] < k:
        raise ValueError(
            f"n_neighbors={k} exceeds filtered reference rows={X_ref.shape[0]}."
        )

    def _run_knn(
        ref_arr: np.ndarray,
        query_arr: np.ndarray,
        *,
        k_use: int,
        tag: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        cache_key = None
        if faiss_cache_prefix:
            cache_key = f"{faiss_cache_prefix}|{tag}"
        idx, dist, meta = knn_search(
            ref_arr,
            query_arr,
            n_neighbors=int(k_use),
            metric=metric,
            backend=knn_backend,
            device=knn_device,
            nprobe=knn_nprobe,
            cache_key=cache_key,
        )
        return (
            np.asarray(idx, dtype=np.int64),
            np.asarray(dist, dtype=np.float32),
            dict(meta),
        )

    if mode == "none":
        return _run_knn(X_ref, X_query, k_use=k, tag="global")

    if ref_tissue is None or query_tissue is None:
        raise ValueError(
            "Hard/soft tissue mode requires tissue annotations in both reference and query."
        )

    ref_tissue = np.asarray(ref_tissue, dtype=object)
    query_tissue = np.asarray(query_tissue, dtype=object)
    n_query = X_query.shape[0]

    if mode == "hard":
        idx_out = np.full((n_query, k), -1, dtype=np.int64)
        dist_out = np.full((n_query, k), np.nan, dtype=np.float32)

        global_idx = None
        global_dist = None
        global_meta = None
        first_meta = None

        for tissue in np.unique(query_tissue):
            q_rows = np.flatnonzero(query_tissue == tissue)
            if q_rows.size == 0:
                continue
            r_rows = np.flatnonzero(ref_tissue == tissue)

            if r_rows.size > 0 and (not hard_allow_global_fallback):
                lk = min(int(k), int(r_rows.size))
                local_idx, local_dist, local_meta = _run_knn(
                    X_ref[r_rows],
                    X_query[q_rows],
                    k_use=int(lk),
                    tag=f"local|{str(tissue)}",
                )
                if first_meta is None:
                    first_meta = local_meta
                mapped = np.full_like(local_idx, -1)
                ok = local_idx >= 0
                if np.any(ok):
                    mapped[ok] = r_rows[local_idx[ok]]
                idx_out[q_rows, :lk] = mapped[:, :lk]
                dist_out[q_rows, :lk] = local_dist[:, :lk]
            elif hard_allow_global_fallback:
                local_min = int(k) if hard_fallback_min_cells is None else max(1, int(hard_fallback_min_cells))
                if r_rows.size >= local_min:
                    lk = min(int(k), int(r_rows.size))
                    local_idx, local_dist, local_meta = _run_knn(
                        X_ref[r_rows],
                        X_query[q_rows],
                        k_use=int(lk),
                        tag=f"local|{str(tissue)}",
                    )
                    if first_meta is None:
                        first_meta = local_meta
                    mapped = np.full_like(local_idx, -1)
                    ok = local_idx >= 0
                    if np.any(ok):
                        mapped[ok] = r_rows[local_idx[ok]]
                    idx_out[q_rows, :lk] = mapped[:, :lk]
                    dist_out[q_rows, :lk] = local_dist[:, :lk]
                else:
                    if global_idx is None or global_dist is None:
                        global_idx, global_dist, global_meta = _run_knn(
                            X_ref,
                            X_query,
                            k_use=k,
                            tag="global_fallback",
                        )
                    idx_out[q_rows] = global_idx[q_rows]
                    dist_out[q_rows] = global_dist[q_rows]
            else:
                # strict hard mode: keep these rows as -1 / NaN when no same-tissue refs
                pass

        knn_meta = global_meta or first_meta or {
            "backend_requested": knn_backend,
            "device_requested": knn_device,
            "backend_used": "sklearn",
            "device_used": "cpu",
        }
        return idx_out, dist_out, knn_meta

    # mode == "soft"
    idx_out = np.full((n_query, k), -1, dtype=np.int64)
    dist_out = np.full((n_query, k), np.nan, dtype=np.float32)
    global_k = min(max(k * 3, k + 5), X_ref.shape[0])

    g_idx, g_dist, g_meta = _run_knn(
        X_ref,
        X_query,
        k_use=int(global_k),
        tag="global_soft",
    )

    tissue_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    first_local_meta = None
    for tissue in np.unique(query_tissue):
        r_rows = np.flatnonzero(ref_tissue == tissue)
        if r_rows.size == 0:
            continue
        q_rows = np.flatnonzero(query_tissue == tissue)
        if q_rows.size == 0:
            continue
        lk = min(k, int(r_rows.size))
        li, ld, lm = _run_knn(
            X_ref[r_rows],
            X_query[q_rows],
            k_use=int(lk),
            tag=f"local|{str(tissue)}",
        )
        if first_local_meta is None:
            first_local_meta = lm
        mapped = np.full_like(li, -1)
        ok = li >= 0
        if np.any(ok):
            mapped[ok] = r_rows[li[ok]]
        tissue_cache[str(tissue)] = (mapped, ld)

    for tissue in np.unique(query_tissue):
        q_rows = np.flatnonzero(query_tissue == tissue)
        if q_rows.size == 0:
            continue
        local_pair = tissue_cache.get(str(tissue), None)

        for local_i, qrow in enumerate(q_rows):
            cand_idx: list[int] = []
            cand_dist: list[float] = []

            for ridx, d in zip(g_idx[qrow], g_dist[qrow]):
                if ridx < 0 or not np.isfinite(d):
                    continue
                penalty = 0.0 if ref_tissue[ridx] == tissue else float(tissue_penalty_lambda)
                cand_idx.append(int(ridx))
                cand_dist.append(float(d + penalty))

            if local_pair is not None:
                li, ld = local_pair
                for ridx, d in zip(li[local_i], ld[local_i]):
                    if ridx < 0 or not np.isfinite(d):
                        continue
                    cand_idx.append(int(ridx))
                    cand_dist.append(float(d))

            if not cand_idx:
                continue

            best: dict[int, float] = {}
            for ridx, d in zip(cand_idx, cand_dist):
                best[ridx] = min(best.get(ridx, np.inf), float(d))
            uniq_idx = np.fromiter(best.keys(), dtype=np.int64)
            uniq_dist = np.fromiter(best.values(), dtype=np.float32)
            order = np.argsort(uniq_dist)[:k]
            k_use = int(order.size)
            idx_out[qrow, :k_use] = uniq_idx[order]
            dist_out[qrow, :k_use] = uniq_dist[order]

    knn_meta = g_meta or first_local_meta or {
        "backend_requested": knn_backend,
        "device_requested": knn_device,
        "backend_used": "sklearn",
        "device_used": "cpu",
    }
    return idx_out, dist_out, knn_meta


# ================================================================
#  1. Predict labels via filtered kNN or "tissue-aware" kNN
# ================================================================

def predict_labels_kNN(
    adata_query,
    adata_ref,
    *,
    # --- Decoupled label config ---
    ref_label_col: str,              # labels used for voting in the REFERENCE
    label_space: str | None = None,  # namespace for outputs/uns keys; defaults to ref_label_col
    query_truth_col: str | None = None,  # optional ground-truth column in QUERY for evaluation

    ref_basis: str = 'X_pca_harmony',
    query_basis: str = 'X_pca_harmony',
    label_suffix: str | None = None,
    time_labels: str = 'time_id',
    n_neighbors: int = 25,
    metric: str = 'cosine',
    knn_backend: str = 'auto',     # "auto" | "faiss" | "sklearn"
    knn_device: str = 'auto',      # "auto" | "cpu" | "cuda" | "cuda:N"
    knn_nprobe: int | None = None, # FAISS IVF nprobe
    omit_labels: list[str] | None = ['unknown','nan','unassigned'],

    # Balancing:
    class_balance: str | None = None,     # None | "global_inverse"
    time_balance: str | None = None,      # None | "global_inverse"
    balance_gamma: float = 1,
    balance_eps: float = 1e-9,

    # Vote distance-weighting (label transfer):
    vote_weighting: str | None = 'gaussian',  # None | "gaussian" | "inverse"
    vote_sigma: float | None = None,          # if None -> per-cell median neighbor distance

    # Time aggregation:
    time_stat_function: str = 'trimmed_mean',  # 'median' | 'mean' | 'trimmed_mean' | 'winsor_mean'
    time_trim_alpha: float = 0.25,
    time_winsor_alpha: float = 0.25,

    # Time distance-weighting:
    time_distance: str | None = 'gaussian',  # None | "gaussian" | "inverse"
    time_sigma: float | None = None,         # if None -> per-cell median neighbor distance
    time_inv_eps: float = 1e-6,
    time_inv_power: float = 1.0,

    # Evaluation
    evaluate: bool = False,
    plot_eval_curves: bool = False,
    plot_mapping_qc: bool = True,
    save_mapping_qc: bool = True,
    show_qc_plots: bool = True,

    # QC thresholds
    p_thresh: float | None = 0.8,
    d_thresh: float | None = None,        # deprecated; use vote_weighting instead
    min_cells_per_label: int = 15,
    apply_filters: bool = True,

    # Output location
    output_dir: str = "zmap_predict",
    # Internal cache guard: keep normal/tissue-aware neighbor caches separated.
    expected_cache_mode: str = "none",
):
    """
    Transfer cell-type labels from a reference to a query dataset using kNN voting.

    Builds a kNN index over the reference embedding, votes on labels using
    distance-weighted nearest neighbors, and writes per-cell predictions and
    confidence scores into ``adata_query.obs``. Reference cells with excluded
    labels (``omit_labels``) are removed from the index *before* building it,
    ensuring clean 1/k probability steps in the vote tallies.

    Results are stored under ``adata_query.uns['zmap_labels'][label_space]``.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Query dataset to annotate.
    adata_ref : anndata.AnnData
        Reference dataset providing labels and the embedding basis.
    ref_label_col : str
        Column in ``adata_ref.obs`` containing the labels to transfer.
    label_space : str or None, default ``None``
        Namespace used for output columns and ``uns`` keys. Defaults to
        ``ref_label_col`` when ``None``.
    query_truth_col : str or None, default ``None``
        Optional ground-truth label column in ``adata_query.obs`` used for
        evaluation metrics when ``evaluate=True``.
    ref_basis : str, default ``"X_pca_harmony"``
        ``obsm`` key in ``adata_ref`` containing the reference embedding.
    query_basis : str, default ``"X_pca_harmony"``
        ``obsm`` key in ``adata_query`` containing the query embedding.
    label_suffix : str or None, default ``None``
        Suffix appended to the predicted label column name in ``adata_query.obs``.
    time_labels : str, default ``"time_id"``
        Column in ``adata_ref.obs`` containing numeric developmental time values
        for time-score aggregation.
    n_neighbors : int, default ``25``
        Number of nearest neighbors used for voting.
    metric : str, default ``"cosine"``
        Distance metric for the kNN index. Passed directly to the underlying
        nearest-neighbor library.
    omit_labels : list of str or None, default ``['unknown', 'nan', 'unassigned']``
        Labels in ``ref_label_col`` to exclude from the kNN index entirely.
        Cells carrying these labels are removed before index construction.
    class_balance : str or None, default ``None``
        Strategy for reweighting votes by class frequency. ``None`` applies no
        reweighting; ``"global_inverse"`` upweights underrepresented classes.
    time_balance : str or None, default ``None``
        Strategy for reweighting votes by time-point frequency. Options mirror
        ``class_balance``.
    balance_gamma : float, default ``1``
        Exponent applied to inverse-frequency weights. Higher values increase
        the strength of balancing.
    vote_weighting : str or None, default ``"gaussian"``
        Distance weighting scheme applied to neighbor votes during label
        transfer. ``None`` uses uniform 1/k voting (discrete probabilities);
        ``"gaussian"`` applies a Gaussian kernel (continuous probabilities,
        recommended); ``"inverse"`` uses inverse-distance weights.
        Gaussian weighting produces better-calibrated confidence scores,
        smoother ROC/PR curves, and makes ``d_thresh`` unnecessary.
    vote_sigma : float or None, default ``None``
        Bandwidth for the Gaussian kernel when ``vote_weighting="gaussian"``.
        If ``None``, uses the per-cell median neighbor distance (adaptive).
    time_stat_function : str, default ``"trimmed_mean"``
        Aggregation function for predicting a continuous time score per cell.
        One of ``"mean"``, ``"median"``, ``"trimmed_mean"``, ``"winsor_mean"``.
    time_trim_alpha : float, default ``0.25``
        Trim fraction used when ``time_stat_function="trimmed_mean"``.
        Must be in ``[0, 0.5)``.
    time_winsor_alpha : float, default ``0.25``
        Winsorization fraction used when ``time_stat_function="winsor_mean"``.
        Must be in ``[0, 0.5)``.
    time_distance : str or None, default ``"gaussian"``
        Distance weighting scheme applied to neighbors when computing the time
        score. ``None`` uses uniform weights; ``"gaussian"`` applies a Gaussian
        kernel; ``"inverse"`` uses inverse-distance weights.
    time_sigma : float or None, default ``None``
        Bandwidth for the Gaussian kernel. If ``None``, uses the per-cell
        median neighbor distance.
    evaluate : bool, default ``False``
        Compute accuracy and other evaluation metrics against ``query_truth_col``.
        Requires ``query_truth_col`` to be set.
    plot_eval_curves : bool, default ``False``
        Plot confidence-threshold curves when ``evaluate=True``.
    plot_mapping_qc : bool, default ``True``
        Plot per-cell confidence and distance QC distributions after prediction.
    save_mapping_qc : bool, default ``True``
        Save QC plots to ``./zmap/predict/``.
    show_qc_plots : bool, default ``True``
        Call ``plt.show()`` for QC plots. Set to ``False`` when display is
        managed by a higher-level wrapper (e.g. ``annotate_with_zmap``).
    p_thresh : float or None, default ``0.8``
        Minimum vote probability required to assign a label. Cells below this
        threshold are marked as unassigned. With ``vote_weighting="gaussian"``,
        this is the only filter needed.
    d_thresh : float or None, default ``None``
        Deprecated. Maximum allowable mean distance to neighbors. Kept for
        backward compatibility but redundant when ``vote_weighting`` is set,
        as distance information is already incorporated into the vote
        probabilities.
    min_cells_per_label : int, default ``15``
        Minimum number of reference cells a label must have to be included in
        voting. Labels with fewer cells are treated as ``omit_labels``.
    apply_filters : bool, default ``True``
        Apply ``p_thresh`` filter to produce the final predicted label column.
        Set to ``False`` to retain raw predictions.

    Returns
    -------
    None
        Results are written directly into ``adata_query``:

        - ``adata_query.obs[f"{label_space}_predicted"]`` — predicted labels.
        - ``adata_query.obs[f"{label_space}_prob"]``      — top-label vote probability.
        - ``adata_query.obs["ZMAP_time_id_predicted"]``  — predicted developmental time.
        - ``adata_query.uns['zmap_labels'][label_space]`` — full run metadata.
    """

    # ---------- helpers ----------
    def _check_alpha(a: float, name: str):
        if not (0.0 <= float(a) < 0.5):
            raise ValueError(f"{name} must be in [0, 0.5). Got {a}.")

    def _trimmed_mean_1d(x: np.ndarray, alpha: float, w: np.ndarray | None = None) -> float:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        _check_alpha(alpha, "time_trim_alpha")
        order = np.argsort(x)
        x_sorted = x[order]
        n = x_sorted.size
        k = int(np.floor(alpha * n))
        if n - 2*k <= 0:
            return float(np.median(x_sorted))
        x_core = x_sorted[k:n-k]
        if w is None:
            return float(np.mean(x_core))
        w_sorted = (w if w.ndim == 1 else np.asarray(w).ravel())[order]
        w_core = w_sorted[k:n-k]
        sw = w_core.sum()
        return float((x_core * w_core).sum() / (sw + balance_eps))

    def _winsorized_mean_1d(x: np.ndarray, alpha: float, w: np.ndarray | None = None) -> float:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        _check_alpha(alpha, "time_winsor_alpha")
        lo, hi = np.quantile(x, [alpha, 1.0 - alpha])
        xw = np.clip(x, lo, hi)
        if w is None:
            return float(np.mean(xw))
        sw = np.sum(w)
        return float((xw * w).sum() / (sw + balance_eps))

    def _pctiles_series(s):
        if s is None:
            return None
        v = pd.to_numeric(pd.Series(s), errors="coerce").to_numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return {"p5": float(np.percentile(v, 5)),
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95))}

    # ---------- validate ----------
    valid_time_stats = {'median', 'mean', 'trimmed_mean', 'winsor_mean'}
    if time_stat_function not in valid_time_stats:
        raise ValueError(f"time_stat_function must be one of {sorted(valid_time_stats)}.")
    if class_balance not in (None, "global_inverse"):
        raise ValueError("class_balance must be one of {None, 'global_inverse'}.")
    if time_balance not in (None, "global_inverse"):
        raise ValueError("time_balance must be one of {None, 'global_inverse'}.")
    if vote_weighting not in (None, "gaussian", "inverse"):
        raise ValueError("vote_weighting must be one of {None, 'gaussian', 'inverse'}.")
    if time_distance not in (None, "gaussian", "inverse"):
        raise ValueError("time_distance must be one of {None, 'gaussian', 'inverse'}.")
    if d_thresh is not None:
        warnings.warn(
            "d_thresh is deprecated and will be removed in a future version. "
            "Distance information is now incorporated via vote_weighting (default 'gaussian'). "
            "Use p_thresh alone for QC filtering.",
            DeprecationWarning,
            stacklevel=2,
        )
    if knn_backend not in {"auto", "faiss", "sklearn"}:
        raise ValueError("knn_backend must be one of {'auto', 'faiss', 'sklearn'}.")
    if knn_nprobe is not None and int(knn_nprobe) <= 0:
        raise ValueError("knn_nprobe must be positive when provided.")
    if expected_cache_mode not in {"none", "hard", "soft"}:
        raise ValueError("expected_cache_mode must be one of {'none', 'hard', 'soft'}.")
    if balance_gamma < 0:
        raise ValueError("balance_gamma must be >= 0.")

    # ---------- namespaces & suffix ----------
    space = label_space or ref_label_col
    if evaluate:
        label_suffix = 'predicted'

    def base_col(lbl: str) -> str:
        return f"{lbl}_{label_suffix}" if (label_suffix is not None and str(label_suffix) != "") else f"{lbl}"

    labels_base = base_col(space)
    # Time label namespace: always "ZMAP_<time_labels>" regardless of main label_space
    if (time_labels is not None) and (time_labels in adata_ref.obs.columns):
        time_ns = f"ZMAP_{time_labels}"        # e.g. "ZMAP_time_id"
        time_base = base_col(time_ns)          # applies suffix like _unfilt or _predicted
    else:
        time_base = None

    # ---------- reference filtering (omit BEFORE kNN) ----------
    if ref_label_col not in adata_ref.obs:
        raise KeyError(f"ref_label_col '{ref_label_col}' not found in adata_ref.obs")

    ref_labels_full = adata_ref.obs[ref_label_col].astype(object)

    # keep only rows with non-missing labels and not in omit_labels
    if omit_labels:
        ref_keep_mask = ~(ref_labels_full.isna() | ref_labels_full.isin(omit_labels))
    else:
        ref_keep_mask = ~ref_labels_full.isna()

    n_ref_keep = int(ref_keep_mask.sum())
    if n_ref_keep < n_neighbors:
        raise ValueError(
            f"After excluding omit_labels/NaNs, only {n_ref_keep} reference cells remain, "
            f"which is fewer than n_neighbors={n_neighbors}. Reduce n_neighbors or relax filtering."
        )

    # compact digest to validate cache
    _mask_digest = [str(int(n_ref_keep)), str(int(np.sum(np.flatnonzero(ref_keep_mask.to_numpy()) % 1048573)))]

    # filtered reference arrays
    X_ref_all = adata_ref.obsm[ref_basis]
    X_ref = X_ref_all[ref_keep_mask.values, :]
    ref_labels = ref_labels_full[ref_keep_mask].astype(object)

    # time (filtered)
    if time_base is not None:
        ref_time_full = pd.to_numeric(adata_ref.obs[time_labels], errors='coerce')
        ref_time = ref_time_full[ref_keep_mask].to_numpy()
    else:
        ref_time = None

    # ---------- kNN graph (cache respects filter) ----------
    reuse_neighbors = False
    knn_meta = {
        "backend_requested": knn_backend,
        "device_requested": knn_device,
        "backend_used": "sklearn",
        "device_used": "cpu",
    }
    if 'zmap_neighbors' in adata_query.uns:
        cache = adata_query.uns['zmap_neighbors']
        same_config = (
            cache.get('ref_basis') == ref_basis and
            cache.get('query_basis') == query_basis and
            cache.get('metric') == metric and
            cache.get('n_neighbors') == n_neighbors and
            list(cache.get('mask_digest', [])) == list(_mask_digest) and
            cache.get('tissue_mode', 'none') == expected_cache_mode and
            cache.get('knn_backend_requested', 'auto') == knn_backend and
            cache.get('knn_device_requested', 'auto') == knn_device and
            cache.get('knn_nprobe_requested', None) == (None if knn_nprobe is None else int(knn_nprobe))
        )
        if same_config:
            _zlog("Reusing cached neighbor graph from adata_query.uns['zmap_neighbors'] (filtered).")
            neighbor_indices = cache['indices']
            distances = cache['distances']
            knn_meta = {
                "backend_requested": cache.get("knn_backend_requested", knn_backend),
                "device_requested": cache.get("knn_device_requested", knn_device),
                "backend_used": cache.get("knn_backend_used", "sklearn"),
                "device_used": cache.get("knn_device_used", "cpu"),
            }
            reuse_neighbors = True

    if not reuse_neighbors:
        _zlog("Computing new kNN graph on filtered reference...")
        neighbor_indices, distances, knn_meta = knn_search(
            X_ref,
            adata_query.obsm[query_basis],
            n_neighbors=n_neighbors,
            metric=metric,
            backend=knn_backend,
            device=knn_device,
            nprobe=knn_nprobe,
        )
        _zlog(
            "kNN backend: requested={req}/{dev}, used={used}/{udev}".format(
                req=knn_meta.get("backend_requested", knn_backend),
                dev=knn_meta.get("device_requested", knn_device),
                used=knn_meta.get("backend_used", "sklearn"),
                udev=knn_meta.get("device_used", "cpu"),
            )
        )
        adata_query.uns['zmap_neighbors'] = {
            'indices': neighbor_indices,
            'distances': distances,
            'ref_basis': ref_basis,
            'query_basis': query_basis,
            'metric': metric,
            'n_neighbors': n_neighbors,
            'mask_digest': _mask_digest,
            'tissue_mode': expected_cache_mode,
            'knn_backend_requested': knn_backend,
            'knn_device_requested': knn_device,
            'knn_nprobe_requested': (None if knn_nprobe is None else int(knn_nprobe)),
            'knn_backend_used': knn_meta.get("backend_used", "sklearn"),
            'knn_device_used': knn_meta.get("device_used", "cpu"),
        }
        _zlog("Cached neighbor graph in adata_query.uns['zmap_neighbors'].")

    # ---------- classes & priors (from filtered ref) ----------
    sorted_classes = np.sort(pd.Series(ref_labels).dropna().astype(str).unique())
    if len(sorted_classes) == 0:
        raise ValueError("No classes remain after filtering; adjust omit_labels or check reference labels.")
    class_indices = {cls: idx for idx, cls in enumerate(sorted_classes)}
    C = len(sorted_classes)

    ref_counts_series = pd.Series(ref_labels).dropna().astype(str).value_counts()
    ref_counts = ref_counts_series.reindex(sorted_classes, fill_value=0).to_numpy(dtype=float)
    ref_total = float(ref_counts.sum()) if ref_counts.sum() > 0 else 1.0
    ref_priors = ref_counts / ref_total

    def _inv_prior(priors: np.ndarray) -> np.ndarray:
        w = np.power(priors + balance_eps, -balance_gamma)
        return w / (w.mean() + balance_eps)

    w_class = _inv_prior(ref_priors) if class_balance == "global_inverse" else np.ones(C, dtype=float)
    w_time_class = _inv_prior(ref_priors) if time_balance == "global_inverse" else np.ones(C, dtype=float)

    # ---------- voting ----------
    ref_labels_values = ref_labels.to_numpy()
    neighbor_classes = ref_labels_values[neighbor_indices]  # shape: (n_query, k)
    probabilities_sorted = np.zeros((neighbor_indices.shape[0], C), dtype=float)
    has_votes = np.zeros(neighbor_indices.shape[0], dtype=bool)

    for i, classes in enumerate(neighbor_classes):
        valid_nbr = (neighbor_indices[i] >= 0) & np.isfinite(distances[i])
        mask = (~pd.isna(classes)) & valid_nbr
        if not np.any(mask):
            continue
        vals = np.asarray(classes[mask], dtype=str)
        idxs = np.searchsorted(sorted_classes, vals)
        di = distances[i][mask]

        # --- distance weighting for label votes ---
        if vote_weighting is None:
            w_vote = np.ones(mask.sum(), dtype=float)
        elif vote_weighting == "gaussian":
            sigma = (vote_sigma if (vote_sigma is not None and vote_sigma > 0)
                     else (np.median(di) + balance_eps))
            w_vote = np.exp(-(di * di) / (2.0 * sigma * sigma))
        else:  # "inverse"
            w_vote = 1.0 / np.power(di + time_inv_eps, time_inv_power)

        # --- class balance weighting ---
        if class_balance == "global_inverse":
            w_vote = w_vote * w_class[idxs]

        scores = np.bincount(idxs, weights=w_vote, minlength=C)
        s = scores.sum()
        if s > 0:
            probabilities_sorted[i, :] = scores / s
            has_votes[i] = True

    predicted_labels = sorted_classes[np.argmax(probabilities_sorted, axis=1)]
    predicted_labels = predicted_labels.astype(object)
    predicted_labels[~has_votes] = np.nan

    # ---------- outputs ----------
    if omit_labels:
        adata_query.uns.setdefault('zmap_labels', {}).setdefault(space, {})
        adata_query.uns['zmap_labels'][space]['Omitted Labels'] = list(omit_labels)

    col_main      = labels_base
    col_unfilt    = f"{labels_base}_unfilt"
    col_prob      = f"{labels_base}_prob"
    col_dist      = f"{labels_base}_dist"
    col_reason    = f"{labels_base}_reason"
    col_reject    = f"{labels_base}_reject_flag"
    col_rareflag  = f"{labels_base}_rare_flag"
    col_probs_mat = f"{labels_base}_probabilities"

    adata_query.obs[col_unfilt] = predicted_labels
    adata_query.obs[col_main]   = adata_query.obs[col_unfilt].copy()
    adata_query.obs[col_prob]   = probabilities_sorted.max(axis=1)
    adata_query.obsm[col_probs_mat] = probabilities_sorted

    # distances (median)
    adata_query.obs[col_dist] = np.median(distances, axis=1)

    # ---------- time transfer (uses filtered ref) ----------
    predicted_time_labels = None
    if time_base is not None:
        # class index per filtered ref cell (for time_balance)
        def _map_class_idx(x):
            if x is None or (isinstance(x, float) and np.isnan(x)): return -1
            return class_indices.get(str(x), -1)
        ref_cls_idx = np.array([_map_class_idx(v) for v in ref_labels.to_numpy()], dtype=int)

        out = np.empty(neighbor_indices.shape[0], dtype=float)
        for i, nbrs in enumerate(neighbor_indices):
            if ref_time is None:
                out[i] = np.nan
                continue

            t = ref_time[nbrs]
            di = distances[i]
            ok = np.isfinite(t) & np.isfinite(di)
            if not np.any(ok):
                out[i] = np.nan
                continue
            t = t[ok]
            di = di[ok]

            if time_balance == "global_inverse":
                ci = ref_cls_idx[nbrs][ok]
                w_class_local = np.where(ci >= 0, w_time_class[ci], 0.0)
            else:
                w_class_local = 1.0

            if time_distance is None:
                w_dist = 1.0
            elif time_distance == "gaussian":
                sigma = (time_sigma if (time_sigma is not None and time_sigma > 0)
                         else (np.median(di) + balance_eps))
                w_dist = np.exp(-(di * di) / (2.0 * sigma * sigma))
            else:  # "inverse"
                w_dist = 1.0 / np.power(di + time_inv_eps, time_inv_power)

            if np.isscalar(w_class_local):
                w = w_dist if np.isscalar(w_dist) else w_dist
            else:
                w = w_class_local * (w_dist if np.isscalar(w_dist) else w_dist)
            if np.isscalar(w):
                w = np.full_like(t, float(w))
            if not np.isfinite(w).any() or np.all(w == 0):
                w = None

            if time_stat_function == 'median' or w is None:
                if time_stat_function == 'median':
                    out[i] = float(np.median(t))
                elif time_stat_function == 'mean':
                    out[i] = float(np.mean(t))
                elif time_stat_function == 'trimmed_mean':
                    out[i] = _trimmed_mean_1d(t, alpha=time_trim_alpha, w=None)
                elif time_stat_function == 'winsor_mean':
                    out[i] = _winsorized_mean_1d(t, alpha=time_winsor_alpha, w=None)
            else:
                if time_stat_function == 'mean':
                    out[i] = float((t * w).sum() / (w.sum() + balance_eps))
                elif time_stat_function == 'trimmed_mean':
                    out[i] = _trimmed_mean_1d(t, alpha=time_trim_alpha, w=w)
                elif time_stat_function == 'winsor_mean':
                    out[i] = _winsorized_mean_1d(t, alpha=time_winsor_alpha, w=w)
                else:
                    out[i] = float(np.median(t))

        predicted_time_labels = out
        adata_query.obs[f"{time_base}_unfilt"] = predicted_time_labels

    _zlog("Predictions complete.")

    # =======================
    #     QC FILTERING
    # =======================
    accept = pd.Series(True, index=adata_query.obs.index)
    if apply_filters:
        use_prob = (p_thresh is not None)
        use_dist = (d_thresh is not None)  # legacy; deprecated

        if not use_prob and not use_dist:
            accept = pd.Series(True, index=adata_query.obs.index)
            _zlog("QC skipped: p_thresh=None → accepting all cells.")
        else:
            p_ok = (adata_query.obs[col_prob] >= p_thresh).fillna(False) if use_prob else pd.Series(True, index=adata_query.obs.index)

            if use_dist:
                # Legacy d_thresh path: OR logic preserved for backward compat
                d_ok = (adata_query.obs[col_dist] <= d_thresh).fillna(False)
                accept = p_ok | d_ok
                _zlog(f"QC applied: p_thresh={p_thresh} OR d_thresh={d_thresh} (d_thresh is deprecated).")
            else:
                # Standard path: single probability gate
                accept = p_ok
                _zlog(f"QC applied: p_thresh={p_thresh}.")

        adata_query.obs[col_reject] = ~accept

        # Mask rejected predictions
        adata_query.obs.loc[~accept, col_main] = np.nan

        n_total = len(accept)
        n_accept = int(accept.sum())
        _zlog(f"{n_accept} accepted / {n_total} total ({n_total - n_accept} rejected).")

        # ---------- QC PLOTTING & OPTIONAL SAVE ----------
        if plot_mapping_qc:
            import os

            os.makedirs(output_dir, exist_ok=True)

            fig_qc, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

            # ----- Probability histogram -----
            ax1.hist(adata_query.obs[col_prob].dropna(), bins=100,
                     color='steelblue', alpha=0.7)
            if use_prob:
                ax1.axvline(p_thresh, color='red', linestyle='--',
                            label=f'p_thresh={p_thresh}')
            ax1.set_title(f"Predicted Probability\n{n_accept} pass / {n_total} total")
            ax1.set_xlabel('Predicted Probability')
            ax1.set_ylabel('Cell Count')
            if ax1.get_legend_handles_labels()[1]:
                ax1.legend()

            # ----- Distance histogram -----
            ax2.hist(adata_query.obs[col_dist].dropna(), bins=100,
                     color='steelblue', alpha=0.7)
            if use_dist:
                ax2.axvline(d_thresh, color='red', linestyle='--',
                            label=f'd_thresh={d_thresh}')
            ax2.set_title(f"{metric.title()} Median Distance (diagnostic)"
                         if not use_dist else
                         f"{metric.title()} Median Distance\n{n_accept} pass / {n_total} total")
            ax2.set_xlabel('Neighbor Distance')
            ax2.set_ylabel('Cell Count')
            if ax2.get_legend_handles_labels()[1]:
                ax2.legend()

            fig_qc.tight_layout()

            # save combined figure
            if save_mapping_qc:
                qc_path = os.path.join(output_dir, f"{labels_base}_qc_summary.png")
                fig_qc.savefig(qc_path, dpi=300, bbox_inches="tight")
                _zlog(f"Saved QC plot: {qc_path}")

            if show_qc_plots:
                plt.show()
            else:
                plt.close(fig_qc)


    # ---------- rare label filter ----------
    if min_cells_per_label is not None and min_cells_per_label > 0:
        _zlog(f"Filtering labels with fewer than {min_cells_per_label} assigned cells...")
        label_counts = adata_query.obs[col_main].value_counts(dropna=True)
        rare_labels = label_counts[label_counts < min_cells_per_label].index
        if len(rare_labels) > 0:
            adata_query.obs[col_rareflag] = adata_query.obs[col_main].isin(rare_labels)
            adata_query.obs.loc[adata_query.obs[col_rareflag], col_main] = np.nan
            adata_query.uns.setdefault('zmap_labels', {}).setdefault(space, {})
            adata_query.uns['zmap_labels'][space]['Rare Labels'] = list(rare_labels)
            _zlog(f"Filtered {len(rare_labels)} rare labels: {list(rare_labels[:10])}{'...' if len(rare_labels) > 10 else ''}")

    # ---------- final time assignment ----------
    if time_base is not None and predicted_time_labels is not None:
        if apply_filters:
            keep_mask = ~adata_query.obs[col_main].isna()
            adata_query.obs[time_base] = np.nan
            adata_query.obs.loc[keep_mask, time_base] = predicted_time_labels[keep_mask.values]
        else:
            adata_query.obs[time_base] = predicted_time_labels

    # ---------- run summary ----------
    try:
        basis_dim = int(adata_ref.obsm[ref_basis].shape[1]) if ref_basis in adata_ref.obsm else None
    except Exception:
        basis_dim = None

    n_total = int(adata_query.n_obs)
    assigned_mask = (~adata_query.obs[col_main].isna()) if col_main in adata_query.obs else pd.Series(False, index=adata_query.obs.index)
    n_assigned = int(assigned_mask.sum())
    pct_assigned = round(100.0 * (n_assigned / n_total) if n_total else 0.0, 2)

    rejection_breakdown = None
    if col_reason in adata_query.obs:
        if col_reject in adata_query.obs:
            rej_mask = adata_query.obs[col_reject].fillna(True).astype(bool)
        else:
            rej_mask = ~assigned_mask
        reasons = adata_query.obs.loc[rej_mask, col_reason]
        if hasattr(reasons, "value_counts"):
            vc = reasons.astype(str).value_counts()
            if len(vc):
                rejection_breakdown = vc.to_dict()

    rare_info = None
    try:
        rare_labels_list = adata_query.uns.get('zmap_labels', {}).get(space, {}).get('Rare Labels', [])
        if rare_labels_list is not None:
            rare_info = {"n_rare_labels_filtered": len(rare_labels_list), "labels": list(rare_labels_list[:10])}
    except Exception:
        pass

    run_summary = {
        "Data": {
            "query_n_cells": int(adata_query.n_obs),
            "ref_n_cells": int(adata_ref.n_obs),
            "ref_basis": ref_basis,
            "query_basis": query_basis,
            "basis_dim": basis_dim,
            "ref_label_col": ref_label_col,
            "query_truth_col": query_truth_col,
            "label_space": space,
            "omit_labels": list(omit_labels or []),
            "classes_ref_total": int(len(sorted_classes)),
            "classes_predicted_total": int(adata_query.obs[col_main].dropna().astype(str).nunique()) if col_main in adata_query.obs else None,
        },
        "Params": {
            "n_neighbors": n_neighbors,
            "metric": metric,
            "knn_backend_requested": knn_meta.get("backend_requested", knn_backend),
            "knn_device_requested": knn_meta.get("device_requested", knn_device),
            "knn_nprobe_requested": (None if knn_nprobe is None else int(knn_nprobe)),
            "knn_backend_used": knn_meta.get("backend_used", "sklearn"),
            "knn_device_used": knn_meta.get("device_used", "cpu"),
            "class_balance": class_balance,
            "time_balance": time_balance,
            "balance_gamma": balance_gamma,
            "balance_eps": balance_eps,
            "vote_weighting": vote_weighting,
            "vote_sigma": ("per-cell-median" if (vote_weighting == "gaussian" and vote_sigma is None) else vote_sigma),
            "time_stat_function": time_stat_function,
            "time_trim_alpha": time_trim_alpha,
            "time_winsor_alpha": time_winsor_alpha,
            "time_distance": time_distance,
            "time_sigma": ("per-cell-median" if (time_distance == "gaussian" and time_sigma is None) else time_sigma),
            "time_inv_power": time_inv_power,
            "time_inv_eps": time_inv_eps,
            "p_thresh": p_thresh,
            "d_thresh": d_thresh,
            "min_cells_per_label": min_cells_per_label,
            "apply_filters": bool(apply_filters),
            "cache_reused": bool(reuse_neighbors),
        },
        "Diagnostics": {
            "probability_summary_unfiltered": _pctiles_series(adata_query.obs[col_prob]) if col_prob in adata_query.obs else None,
            "neighbor_distance_summary": _pctiles_series(adata_query.obs[col_dist]) if col_dist in adata_query.obs else None,
        },
        "Coverage": {
            "n_total": n_total,
            "n_assigned": n_assigned,
            "pct_assigned": pct_assigned,
            "n_rejected": n_total - n_assigned,
            "pct_rejected": round(100.0 - pct_assigned, 2),
            "rejection_breakdown": rejection_breakdown,
            "rare_label_filter": rare_info,
        },
    }

    adata_query.uns.setdefault('zmap_labels', {}).setdefault(space, {})
    adata_query.uns['zmap_labels'][space]["Run Summary"] = run_summary

    # ---------- evaluation ----------
    if evaluate:
        if not query_truth_col or query_truth_col not in adata_query.obs.columns:
            _zlog(f"Evaluation skipped: ground-truth column '{query_truth_col}' not found in adata_query.obs.")
            _zlog(f"Finished predicting and annotating: {space}")
            return

        _zlog("Evaluating model performance on ACCEPTED predictions only...")
        has_truth = ~adata_query.obs[query_truth_col].isna()
        not_rejected = (~adata_query.obs[col_reject].fillna(True)) if (apply_filters and col_reject in adata_query.obs) else True
        has_pred = ~adata_query.obs[col_main].isna()
        eval_mask = has_truth & not_rejected & has_pred

        n_eval = int(eval_mask.sum())
        if n_eval == 0:
            _zlog("No accepted rows available for evaluation after filtering; metrics not computed.")
            _zlog(f"Finished predicting and annotating: {space}")
            return

        true_labels_values      = adata_query.obs.loc[eval_mask, query_truth_col].astype(str).values
        predicted_labels_values = adata_query.obs.loc[eval_mask, col_main].astype(str).values
        probabilities_eval      = adata_query.obsm[col_probs_mat][eval_mask, :]

        true_classes = set(np.unique(true_labels_values))
        predicted_classes = set(np.unique(predicted_labels_values))
        overlapping_classes = sorted(true_classes.intersection(predicted_classes))
        if len(overlapping_classes) == 0:
            _zlog("No overlapping classes between true and predicted after filtering; metrics not computed.")
            _zlog(f"Finished predicting and annotating: {space}")
            return

        y_true_binarized = label_binarize(true_labels_values, classes=overlapping_classes)
        col_idx = [class_indices[cls] for cls in overlapping_classes]
        probabilities_eval = probabilities_eval[:, col_idx]

        per_class = precision_recall_fscore_support(
            true_labels_values, predicted_labels_values, labels=overlapping_classes, zero_division=0
        )
        cm = confusion_matrix(true_labels_values, predicted_labels_values, labels=overlapping_classes)
        cm_df = pd.DataFrame(cm, index=overlapping_classes, columns=overlapping_classes)

        accuracy        = accuracy_score(true_labels_values, predicted_labels_values)
        macro_precision = precision_score(true_labels_values, predicted_labels_values, average='macro', zero_division=0)
        macro_recall    = recall_score(true_labels_values, predicted_labels_values, average='macro', zero_division=0)
        macro_f1        = f1_score(true_labels_values, predicted_labels_values, average='macro', zero_division=0)

        class_auroc = {}
        for i, label in enumerate(overlapping_classes):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], probabilities_eval[:, i])
            class_auroc[label] = auc(fpr, tpr)
        macro_auroc = roc_auc_score(y_true_binarized, probabilities_eval, average='macro')

        df_aggregate = pd.DataFrame({
            "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro AUROC"],
            "Score": [accuracy, macro_precision, macro_recall, macro_f1, macro_auroc],
        })
        df_per_class = pd.DataFrame({
            "Class": overlapping_classes,
            "Precision": per_class[0],
            "Recall": per_class[1],
            "F1-Score": per_class[2],
            "AUROC": [class_auroc[label] for label in overlapping_classes],
            "Support": per_class[3],
        })

        # Store evaluation results (merge into existing uns, don't overwrite)
        adata_query.uns.setdefault('zmap_labels', {}).setdefault(space, {})
        adata_query.uns['zmap_labels'][space]["Evaluation"] = {
            "Aggregate Metrics": df_aggregate,
            "Class-Specific Metrics": df_per_class,
            "Confusion Matrix": cm_df,
            "Eval N": n_eval,
        }

        _zlog(f"Evaluation complete ({n_eval:,} cells):")
        _display_df(df_aggregate)

        # ---- Evaluation output directory ----
        eval_dir = os.path.join(output_dir, "evaluation")
        if save_mapping_qc:
            os.makedirs(eval_dir, exist_ok=True)
            # Save metric tables
            df_aggregate.to_csv(os.path.join(eval_dir, f"{labels_base}_eval_aggregate.csv"), index=False)
            df_per_class.to_csv(os.path.join(eval_dir, f"{labels_base}_eval_per_class.csv"), index=False)
            cm_df.to_csv(os.path.join(eval_dir, f"{labels_base}_eval_confusion_matrix.csv"))
            _zlog(f"Saved evaluation tables → {eval_dir}/")

        # ---- Plot ROC / PR curves ----
        if plot_eval_curves:
            _zlog("Plotting ROC and PR curves...")
            for i, label in enumerate(overlapping_classes):
                fig_eval, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(8, 4))

                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], probabilities_eval[:, i])
                precision_vals, recall_vals, _ = precision_recall_curve(y_true_binarized[:, i], probabilities_eval[:, i])

                ax_roc.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.2f}")
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_title(f"ROC – {label}")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend()

                ax_pr.plot(recall_vals, precision_vals)
                ax_pr.set_title(f"Precision–Recall – {label}")
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")

                fig_eval.tight_layout()

                if save_mapping_qc:
                    os.makedirs(eval_dir, exist_ok=True)
                    safe_label = label.replace("/", "_").replace(" ", "_")
                    fig_path = os.path.join(eval_dir, f"{labels_base}_eval_{safe_label}.png")
                    fig_eval.savefig(fig_path, dpi=300, bbox_inches="tight")

                if show_qc_plots:
                    plt.show()
                else:
                    plt.close(fig_eval)

            if save_mapping_qc:
                _zlog(f"Saved {len(overlapping_classes)} evaluation figures → {eval_dir}/")

    _zlog(f"Finished predicting and annotating: {space}")


def predict_labels_tissue_kNN(
    adata_query,
    adata_ref,
    *,
    # --- Decoupled label config ---
    ref_label_col: str,
    label_space: str | None = None,
    query_truth_col: str | None = None,
    ref_basis: str = "X_pca_harmony",
    query_basis: str = "X_pca_harmony",
    label_suffix: str | None = None,
    time_labels: str = "time_id",
    n_neighbors: int = 25,
    metric: str = "cosine",
    # --- compatibility aliases (encoder-style names) ---
    ref_latent_key: str | None = None,
    query_latent_key: str | None = None,
    k: int | None = None,
    knn_metric: str | None = None,
    tissue_col: str | None = None,
    # --- tissue-aware controls ---
    tissue_mode: str = "hard",           # "none" | "hard" | "soft"
    ref_tissue_col: str = "ZMAP_Tissue",
    query_tissue_col: str = "ZMAP_Tissue",
    tissue_penalty_lambda: float = 1.0,
    hard_fallback_min_cells: int | None = 10,
    # --- backend controls (aligned with knn_backend.py) ---
    knn_backend: str = "auto",           # "auto" | "faiss" | "sklearn"
    knn_device: str = "auto",            # "auto" | "cpu" | "cuda" | "cuda:N"
    knn_nprobe: int | None = None,
    knn_l2norm: bool = False,
    # --- compatibility params from tissue-aware API ---
    class_prior_alpha: float = 0.0,
    pseudo_tissue_k: int | None = None,
    pseudo_tissue_threshold: float = 0.0,
    pseudo_tissue_margin_threshold: float = 0.0,
    auto_pseudo_tissue: bool = True,
    fallback_to_plain_knn: bool = True,
    pseudo_tissue_unknown_label: str = "unknown",
    reuse_knn_cache: bool = True,
    confidence_threshold: float | None = None,
    margin_threshold: float = 0.0,
    include_unassigned: bool = False,
    run_time_prediction: bool = False,
    time_col: str = "time_group_id",
    time_order: str | list[str] | None = None,
    time_topk: int = 5,
    time_hard_topk: int = 5,
    time_trim_extremes: int = 1,
    time_tau: float = 0.0,
    time_monotone_delta: int = 0,
    time_monotone_gamma: float = 1.0,
    # --- same downstream controls as predict_labels_kNN ---
    omit_labels: list[str] | None = ["unknown", "nan", "unassigned"],
    class_balance: str | None = None,
    time_balance: str | None = None,
    balance_gamma: float = 1,
    balance_eps: float = 1e-9,
    vote_weighting: str | None = "gaussian",
    vote_sigma: float | None = None,
    time_stat_function: str = "trimmed_mean",
    time_trim_alpha: float = 0.25,
    time_winsor_alpha: float = 0.25,
    time_distance: str | None = "gaussian",
    time_sigma: float | None = None,
    time_inv_eps: float = 1e-6,
    time_inv_power: float = 1.0,
    evaluate: bool = False,
    plot_eval_curves: bool = False,
    plot_mapping_qc: bool = True,
    save_mapping_qc: bool = True,
    show_qc_plots: bool = True,
    p_thresh: float | None = 0.8,
    d_thresh: float | None = None,
    min_cells_per_label: int = 15,
    apply_filters: bool = True,
    output_dir: str = "zmap_predict",
):
    """
    Tissue-aware variant of step-3 label transfer.

    This function computes a tissue-aware neighbor graph from the step-2
    embedding (`query_basis`), caches it into `adata_query.uns['zmap_neighbors']`,
    then reuses `predict_labels_kNN(...)` for voting/QC/summary so step-4 inputs
    remain unchanged.
    """
    if ref_latent_key is not None:
        ref_basis = str(ref_latent_key)
    if query_latent_key is not None:
        query_basis = str(query_latent_key)
    if k is not None:
        n_neighbors = int(k)
    if knn_metric is not None:
        metric = str(knn_metric)
    if tissue_col is not None:
        ref_tissue_col = str(tissue_col)
        query_tissue_col = str(tissue_col)

    mode = str(tissue_mode).lower()
    if mode not in {"none", "hard", "soft"}:
        raise ValueError("tissue_mode must be one of {'none', 'hard', 'soft'}.")
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be one of {'euclidean', 'cosine'}.")
    if knn_backend not in {"auto", "faiss", "sklearn"}:
        raise ValueError("knn_backend must be one of {'auto', 'faiss', 'sklearn'}.")
    if knn_nprobe is not None and int(knn_nprobe) <= 0:
        raise ValueError("knn_nprobe must be positive when provided.")
    if int(n_neighbors) <= 0:
        raise ValueError("n_neighbors must be positive.")
    if hard_fallback_min_cells is not None and int(hard_fallback_min_cells) <= 0:
        raise ValueError("hard_fallback_min_cells must be positive when provided.")
    if pseudo_tissue_k is not None and int(pseudo_tissue_k) <= 0:
        raise ValueError("pseudo_tissue_k must be positive when provided.")
    if float(pseudo_tissue_threshold) < 0:
        raise ValueError("pseudo_tissue_threshold must be >= 0.")
    if float(pseudo_tissue_margin_threshold) < 0:
        raise ValueError("pseudo_tissue_margin_threshold must be >= 0.")
    if float(class_prior_alpha) != 0.0:
        _zlog(
            "predict_labels_tissue_kNN: class_prior_alpha is accepted for "
            "API compatibility but not used in predict_labels_kNN voting."
        )
    if bool(run_time_prediction):
        _zlog(
            "predict_labels_tissue_kNN: run_time_prediction/time_* parameters "
            "are accepted for API compatibility. Time transfer is controlled by "
            "predict_labels_kNN(time_labels=...)."
        )

    if ref_basis not in adata_ref.obsm:
        raise KeyError(f"ref_basis '{ref_basis}' not found in adata_ref.obsm")
    if query_basis not in adata_query.obsm:
        raise KeyError(f"query_basis '{query_basis}' not found in adata_query.obsm")
    if ref_label_col not in adata_ref.obs:
        raise KeyError(f"ref_label_col '{ref_label_col}' not found in adata_ref.obs")

    # Align omit behavior with include_unassigned flag from tissue-aware API.
    omit_effective = list(omit_labels or [])
    if include_unassigned:
        omit_effective = [x for x in omit_effective if str(x).lower() != "unassigned"]

    ref_labels_full = adata_ref.obs[ref_label_col].astype(object)
    if omit_effective:
        ref_keep_mask = ~(ref_labels_full.isna() | ref_labels_full.isin(omit_effective))
    else:
        ref_keep_mask = ~ref_labels_full.isna()

    n_ref_keep = int(ref_keep_mask.sum())
    if n_ref_keep < int(n_neighbors):
        raise ValueError(
            f"After excluding omit_labels/NaNs, only {n_ref_keep} reference cells remain, "
            f"which is fewer than n_neighbors={n_neighbors}."
        )

    mask_digest = [
        str(int(n_ref_keep)),
        str(int(np.sum(np.flatnonzero(ref_keep_mask.to_numpy()) % 1048573))),
    ]

    p_thresh_use = p_thresh
    if confidence_threshold is not None:
        p_thresh_use = float(confidence_threshold)
    hard_min_use = (
        None if mode != "hard"
        else (None if hard_fallback_min_cells is None else int(hard_fallback_min_cells))
    )

    space = label_space or ref_label_col
    pseudo_info: dict[str, Any] | None = None
    pseudo_used_for_transfer = False

    def _run_plain_transfer(*, fallback_reason: str | None, pseudo_used: bool) -> None:
        predict_labels_kNN(
            adata_query,
            adata_ref,
            ref_label_col=ref_label_col,
            label_space=label_space,
            query_truth_col=query_truth_col,
            ref_basis=ref_basis,
            query_basis=query_basis,
            label_suffix=label_suffix,
            time_labels=time_labels,
            n_neighbors=int(n_neighbors),
            metric=metric,
            knn_backend=knn_backend,
            knn_device=knn_device,
            knn_nprobe=knn_nprobe,
            omit_labels=omit_effective,
            class_balance=class_balance,
            time_balance=time_balance,
            balance_gamma=balance_gamma,
            balance_eps=balance_eps,
            vote_weighting=vote_weighting,
            vote_sigma=vote_sigma,
            time_stat_function=time_stat_function,
            time_trim_alpha=time_trim_alpha,
            time_winsor_alpha=time_winsor_alpha,
            time_distance=time_distance,
            time_sigma=time_sigma,
            time_inv_eps=time_inv_eps,
            time_inv_power=time_inv_power,
            evaluate=evaluate,
            plot_eval_curves=plot_eval_curves,
            plot_mapping_qc=plot_mapping_qc,
            save_mapping_qc=save_mapping_qc,
            show_qc_plots=show_qc_plots,
            p_thresh=p_thresh_use,
            d_thresh=d_thresh,
            min_cells_per_label=min_cells_per_label,
            apply_filters=apply_filters,
            output_dir=output_dir,
            expected_cache_mode="none",
        )
        adata_query.uns.setdefault("zmap_labels", {}).setdefault(space, {})
        adata_query.uns["zmap_labels"][space]["Tissue-aware kNN"] = {
            "requested_tissue_mode": mode,
            "effective_tissue_mode": "none",
            "fallback_to_plain_knn": True,
            "fallback_reason": fallback_reason,
            "auto_pseudo_tissue": bool(auto_pseudo_tissue),
            "pseudo_tissue_used": bool(pseudo_used),
            "hard_allow_global_fallback": None,
            "hard_fallback_min_cells": hard_min_use,
            "pseudo_tissue_k": (None if pseudo_tissue_k is None else int(pseudo_tissue_k)),
            "pseudo_tissue_threshold": float(pseudo_tissue_threshold),
            "pseudo_tissue_margin_threshold": float(pseudo_tissue_margin_threshold),
            "query_tissue_col": str(query_tissue_col),
            "ref_tissue_col": str(ref_tissue_col),
        }

    if mode == "none":
        _zlog("tissue_mode='none' -> running plain predict_labels_kNN.")
        _run_plain_transfer(fallback_reason="tissue_mode_none", pseudo_used=False)
        return

    # Auto pseudo-tissue when query tissues are missing; otherwise optionally fallback.
    missing_reason = None
    has_ref_tissue = ref_tissue_col in adata_ref.obs
    has_query_tissue = _has_valid_tissue_labels(adata_query, query_tissue_col)
    if not has_ref_tissue:
        missing_reason = f"missing_ref_tissue_col:{ref_tissue_col}"
    elif not has_query_tissue and bool(auto_pseudo_tissue):
        X_ref_pseudo = np.asarray(adata_ref.obsm[ref_basis][ref_keep_mask.values, :], dtype=np.float32)
        X_query_pseudo = np.asarray(adata_query.obsm[query_basis], dtype=np.float32)
        if knn_l2norm:
            X_ref_pseudo = _l2_normalize_rows(X_ref_pseudo)
            X_query_pseudo = _l2_normalize_rows(X_query_pseudo)
        ref_tissue_filtered = adata_ref.obs[ref_tissue_col].astype(str).to_numpy()[ref_keep_mask.values]
        faiss_cache_prefix = (
            f"taware|ref={ref_basis}|qry={query_basis}|n_ref={X_ref_pseudo.shape[0]}|"
            f"metric={metric}|mode={mode}|l2={int(bool(knn_l2norm))}"
        )
        _zlog(
            f"query tissue '{query_tissue_col}' missing; "
            "running pseudo tissue prediction first."
        )
        pseudo_info = _predict_pseudo_tissue_knn(
            adata_query,
            X_ref=X_ref_pseudo,
            X_query=X_query_pseudo,
            ref_tissue=ref_tissue_filtered,
            query_tissue_col=str(query_tissue_col),
            ref_tissue_col=str(ref_tissue_col),
            n_neighbors=int(n_neighbors),
            metric=str(metric),
            knn_backend=str(knn_backend),
            knn_device=str(knn_device),
            knn_nprobe=(None if knn_nprobe is None else int(knn_nprobe)),
            pseudo_tissue_k=(None if pseudo_tissue_k is None else int(pseudo_tissue_k)),
            pseudo_tissue_threshold=float(pseudo_tissue_threshold),
            pseudo_tissue_margin_threshold=float(pseudo_tissue_margin_threshold),
            unknown_label=str(pseudo_tissue_unknown_label),
            pseudo_col=f"{query_tissue_col}_pseudo",
            faiss_cache_prefix=faiss_cache_prefix,
            write_query_tissue_col=True,
            plot_qc=bool(plot_mapping_qc),
            save_qc=bool(save_mapping_qc),
            output_dir=str(output_dir),
        )
        pseudo_used_for_transfer = True
        has_query_tissue = _has_valid_tissue_labels(adata_query, query_tissue_col)
        if not has_query_tissue:
            missing_reason = f"pseudo_tissue_failed:{query_tissue_col}"
    elif not has_query_tissue:
        missing_reason = f"missing_query_tissue_col:{query_tissue_col}"

    if missing_reason is not None:
        if bool(fallback_to_plain_knn):
            _zlog(
                f"Tissue-aware unavailable ({missing_reason}); "
                "falling back to plain predict_labels_kNN."
            )
            _run_plain_transfer(
                fallback_reason=missing_reason,
                pseudo_used=bool(pseudo_used_for_transfer),
            )
            return
        if str(missing_reason).startswith("missing_ref_tissue_col:"):
            raise KeyError(
                f"Missing tissue column in adata_ref.obs: {ref_tissue_col}. "
                "Set fallback_to_plain_knn=True to fallback."
            )
        raise KeyError(
            f"Missing tissue column in adata_query.obs: {query_tissue_col}. "
            "Enable auto_pseudo_tissue=True or set fallback_to_plain_knn=True."
        )

    cache = adata_query.uns.get("zmap_neighbors", {})
    reuse_neighbors = False
    if bool(reuse_knn_cache) and isinstance(cache, dict):
        reuse_neighbors = (
            cache.get("ref_basis") == ref_basis
            and cache.get("query_basis") == query_basis
            and cache.get("metric") == metric
            and cache.get("n_neighbors") == int(n_neighbors)
            and cache.get("mask_digest") == mask_digest
            and cache.get("tissue_mode") == mode
            and cache.get("ref_tissue_col") == ref_tissue_col
            and cache.get("query_tissue_col") == query_tissue_col
            and float(cache.get("tissue_penalty_lambda", tissue_penalty_lambda)) == float(tissue_penalty_lambda)
            and bool(cache.get("hard_allow_global_fallback", True)) is False
            and cache.get("hard_fallback_min_cells", None) == hard_min_use
            and bool(cache.get("knn_l2norm", False)) == bool(knn_l2norm)
            and cache.get("knn_backend_requested", "auto") == knn_backend
            and cache.get("knn_device_requested", "auto") == knn_device
            and cache.get("knn_nprobe_requested", None) == (None if knn_nprobe is None else int(knn_nprobe))
            and "indices" in cache
            and "distances" in cache
        )

    if not reuse_neighbors:
        X_ref = np.asarray(adata_ref.obsm[ref_basis][ref_keep_mask.values, :], dtype=np.float32)
        X_query = np.asarray(adata_query.obsm[query_basis], dtype=np.float32)
        if knn_l2norm:
            X_ref = _l2_normalize_rows(X_ref)
            X_query = _l2_normalize_rows(X_query)

        ref_tissue = adata_ref.obs[ref_tissue_col].astype(str).to_numpy()[ref_keep_mask.values]
        query_tissue = adata_query.obs[query_tissue_col].astype(str).to_numpy()
        faiss_cache_prefix = (
            f"taware|ref={ref_basis}|qry={query_basis}|n_ref={X_ref.shape[0]}|"
            f"metric={metric}|mode={mode}|l2={int(bool(knn_l2norm))}"
        )
        idx, dist, knn_meta = _compute_tissue_aware_neighbors(
            X_ref=X_ref,
            X_query=X_query,
            ref_tissue=ref_tissue,
            query_tissue=query_tissue,
            n_neighbors=int(n_neighbors),
            metric=str(metric),
            tissue_mode=mode,
            tissue_penalty_lambda=float(tissue_penalty_lambda),
            hard_allow_global_fallback=False,
            hard_fallback_min_cells=hard_min_use,
            knn_backend=str(knn_backend),
            knn_device=str(knn_device),
            knn_nprobe=(None if knn_nprobe is None else int(knn_nprobe)),
            faiss_cache_prefix=faiss_cache_prefix,
        )
        adata_query.uns["zmap_neighbors"] = {
            "indices": idx,
            "distances": dist,
            "ref_basis": ref_basis,
            "query_basis": query_basis,
            "metric": metric,
            "n_neighbors": int(n_neighbors),
            "mask_digest": mask_digest,
            "tissue_mode": mode,
            "ref_tissue_col": ref_tissue_col,
            "query_tissue_col": query_tissue_col,
            "tissue_penalty_lambda": float(tissue_penalty_lambda),
            "hard_allow_global_fallback": False,
            "hard_fallback_min_cells": hard_min_use,
            "knn_l2norm": bool(knn_l2norm),
            "knn_backend_requested": knn_meta.get("backend_requested", knn_backend),
            "knn_device_requested": knn_meta.get("device_requested", knn_device),
            "knn_nprobe_requested": (None if knn_nprobe is None else int(knn_nprobe)),
            "knn_backend_used": knn_meta.get("backend_used", "sklearn"),
            "knn_device_used": knn_meta.get("device_used", "cpu"),
        }
    else:
        _zlog("Reusing cached tissue-aware neighbor graph from adata_query.uns['zmap_neighbors'].")

    predict_labels_kNN(
        adata_query,
        adata_ref,
        ref_label_col=ref_label_col,
        label_space=label_space,
        query_truth_col=query_truth_col,
        ref_basis=ref_basis,
        query_basis=query_basis,
        label_suffix=label_suffix,
        time_labels=time_labels,
        n_neighbors=int(n_neighbors),
        metric=metric,
        knn_backend=knn_backend,
        knn_device=knn_device,
        knn_nprobe=knn_nprobe,
        omit_labels=omit_effective,
        class_balance=class_balance,
        time_balance=time_balance,
        balance_gamma=balance_gamma,
        balance_eps=balance_eps,
        vote_weighting=vote_weighting,
        vote_sigma=vote_sigma,
        time_stat_function=time_stat_function,
        time_trim_alpha=time_trim_alpha,
        time_winsor_alpha=time_winsor_alpha,
        time_distance=time_distance,
        time_sigma=time_sigma,
        time_inv_eps=time_inv_eps,
        time_inv_power=time_inv_power,
        evaluate=evaluate,
        plot_eval_curves=plot_eval_curves,
        plot_mapping_qc=plot_mapping_qc,
        save_mapping_qc=save_mapping_qc,
        show_qc_plots=show_qc_plots,
        p_thresh=p_thresh_use,
        d_thresh=d_thresh,
        min_cells_per_label=min_cells_per_label,
        apply_filters=apply_filters,
        output_dir=output_dir,
        expected_cache_mode=mode,
    )

    adata_query.uns.setdefault("zmap_labels", {}).setdefault(space, {})
    adata_query.uns["zmap_labels"][space]["Tissue-aware kNN"] = {
        "requested_tissue_mode": mode,
        "effective_tissue_mode": mode,
        "ref_tissue_col": ref_tissue_col,
        "query_tissue_col": query_tissue_col,
        "tissue_penalty_lambda": float(tissue_penalty_lambda),
        "knn_backend_requested": adata_query.uns.get("zmap_neighbors", {}).get(
            "knn_backend_requested", knn_backend
        ),
        "knn_device_requested": adata_query.uns.get("zmap_neighbors", {}).get(
            "knn_device_requested", knn_device
        ),
        "knn_nprobe_requested": adata_query.uns.get("zmap_neighbors", {}).get(
            "knn_nprobe_requested", (None if knn_nprobe is None else int(knn_nprobe))
        ),
        "knn_backend_used": adata_query.uns.get("zmap_neighbors", {}).get(
            "knn_backend_used", "sklearn"
        ),
        "knn_device_used": adata_query.uns.get("zmap_neighbors", {}).get(
            "knn_device_used", "cpu"
        ),
        "knn_l2norm": bool(knn_l2norm),
        "reuse_knn_cache": bool(reuse_knn_cache),
        "auto_pseudo_tissue": bool(auto_pseudo_tissue),
        "fallback_to_plain_knn": False,
        "fallback_reason": None,
        "pseudo_tissue_used": bool(pseudo_used_for_transfer),
        "hard_allow_global_fallback": (
            None if mode != "hard" else False
        ),
        "hard_fallback_min_cells": hard_min_use,
        "class_prior_alpha": float(class_prior_alpha),
        "pseudo_tissue_k": (None if pseudo_tissue_k is None else int(pseudo_tissue_k)),
        "pseudo_tissue_threshold": float(pseudo_tissue_threshold),
        "pseudo_tissue_margin_threshold": float(pseudo_tissue_margin_threshold),
        "pseudo_tissue_unknown_label": str(pseudo_tissue_unknown_label),
        "pseudo_tissue_col": (None if pseudo_info is None else pseudo_info.get("pseudo_col")),
        "confidence_threshold": (None if confidence_threshold is None else float(confidence_threshold)),
        "margin_threshold": float(margin_threshold),
        "include_unassigned": bool(include_unassigned),
        "run_time_prediction": bool(run_time_prediction),
        "time_col": str(time_col),
        "time_order": time_order,
        "time_topk": int(time_topk),
        "time_hard_topk": int(time_hard_topk),
        "time_trim_extremes": int(time_trim_extremes),
        "time_tau": float(time_tau),
        "time_monotone_delta": int(time_monotone_delta),
        "time_monotone_gamma": float(time_monotone_gamma),
    }


# ================================================================
#  2. Summarize kNN run
# ================================================================

def summarize_knn_run(adata_query, label_key):
    """
    Return a concise summary table for a completed kNN label-transfer run.

    Reads the run metadata stored in
    ``adata_query.uns['zmap_labels'][label_key]`` and formats the key
    statistics as a two-column ``DataFrame``.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Query dataset that has been annotated by ``predict_labels_kNN`` or
        ``annotate_with_zmap``.
    label_key : str
        The ``label_space`` used when the prediction was run (matches the key
        under ``adata_query.uns['zmap_labels']``).

    Returns
    -------
    pd.DataFrame
        Two-column table with columns ``["Key", "Value"]`` containing:

        - ``label_space``   — label namespace used.
        - ``n_neighbors``   — number of neighbors in the kNN run.
        - ``metric``        — distance metric used.
        - ``p_thresh``      — probability threshold applied.
        - ``n_assigned``    — number of cells that received a label.
        - ``pct_assigned``  — percentage of cells that received a label.

    Raises
    ------
    KeyError
        If ``label_key`` is not found in ``adata_query.uns['zmap_labels']``,
        or if the run metadata is missing a ``"Run Summary"`` entry.
    """
    try:
        d = adata_query.uns['zmap_labels'][label_key]
    except KeyError:
        raise KeyError(
            f"Could not find results at adata_query.uns['zmap_labels']['{label_key}']."
        )

    if "Run Summary" not in d:
        raise KeyError("'Run Summary' missing under that label_key.")

    run = d["Run Summary"]
    data = run.get("Data", {})
    params = run.get("Params", {})
    cov = run.get("Coverage", {})

    label_identifier = data.get("true_labels") or data.get("label_space") or label_key

    df = pd.DataFrame([
        ("label_space", label_identifier),
        ("n_neighbors", params.get("n_neighbors")),
        ("metric", params.get("metric")),
        ("knn_backend", params.get("knn_backend_used")),
        ("knn_device", params.get("knn_device_used")),
        ("p_thresh", params.get("p_thresh")),
        ("n_assigned", cov.get("n_assigned")),
        ("pct_assigned", cov.get("pct_assigned")),
    ], columns=["Key", "Value"])

    df["Value"] = df["Value"].astype(str)

    return df


# ================================================================
#  3. Aggregate cell annotations to cluster-level consensus
# ================================================================

def aggregate_by_cluster(
    adata_query: ad.AnnData,
    cluster_col: str,
    label_space: str,
    *,
    save_csv: bool = True,
    output_dir: str = "zmap_predict",
) -> pd.DataFrame:
    """
    Aggregate cell-level ZMAP annotations to cluster-level consensus calls.

    For each cluster in ``cluster_col``, identifies the plurality label among
    all QC-assigned (non-NA) cells, computes the fraction of assigned cells
    carrying that label (consensus fraction), the mean per-cell kNN vote
    probability for those cells, and the margin over the second-ranked label.
    Also reports raw coverage counts so the user can assess per-cluster
    annotation quality (e.g., clusters where most cells were rejected).

    Parameters
    ----------
    adata_query : anndata.AnnData
        Query dataset annotated by ``predict_labels_kNN`` or
        ``annotate_with_zmap``.
    cluster_col : str
        Column in ``adata_query.obs`` containing user-defined cluster IDs
        (e.g. ``"leiden"``).
    label_space : str
        Label namespace used during prediction (must match
        ``adata_query.uns['zmap_labels'][label_space]``). Used to derive
        the predicted-label and probability column names.
    save_csv : bool, default ``True``
        Write the cluster summary table to
        ``./zmap/predict/{label_space}_cluster_summary.csv``.

    Returns
    -------
    pd.DataFrame
        One row per cluster, sorted by cluster ID, with columns:

        - ``cluster``          — cluster identifier.
        - ``n_cells_total``    — total cells in cluster.
        - ``n_cells_assigned`` — cells with a non-NA predicted label (passed QC).
        - ``pct_assigned``     — percentage of cells that passed QC.
        - ``top_label``        — plurality ZMAP label among assigned cells.
        - ``top_fraction``     — fraction of assigned cells carrying the top label.
        - ``mean_prob``        — mean kNN vote probability of top-label cells.
        - ``margin``           — ``top_fraction`` − ``second_fraction``;
          ``NaN`` when fewer than 2 distinct labels are present.
        - ``second_label``     — second-ranked label; ``NaN`` when only one
          label is present.
        - ``second_fraction``  — fraction of second-ranked label; ``NaN`` when
          only one label is present.

    Raises
    ------
    KeyError
        If ``cluster_col`` or the predicted-label column derived from
        ``label_space`` is not found in ``adata_query.obs``.

    Notes
    -----
    The aggregation operates only on cells whose predicted label is non-NA
    (i.e., cells that passed QC filters in ``predict_labels_kNN``). Rejected
    cells are counted in ``n_cells_total`` but excluded from voting, so that
    ``top_fraction`` and ``margin`` reflect the confidence of the *accepted*
    predictions rather than being diluted by noise.

    ``mean_prob`` reflects the mean per-cell kNN vote probability for
    top-label cells only, and is distinct from ``top_fraction``.
    ``top_fraction`` captures cluster-level consensus (how unanimously
    assigned cells agree); ``mean_prob`` captures how confident the kNN
    classifier was for those individual cells.
    """
    labels_base = f"{label_space}_predicted"
    col_main    = labels_base
    col_prob    = f"{labels_base}_prob"

    if cluster_col not in adata_query.obs.columns:
        raise KeyError(
            f"cluster_col '{cluster_col}' not found in adata_query.obs. "
            f"Available columns: {list(adata_query.obs.columns)}"
        )
    if col_main not in adata_query.obs.columns:
        raise KeyError(
            f"Predicted label column '{col_main}' not found in adata_query.obs. "
            f"Run predict_labels_kNN with label_space='{label_space}' first."
        )

    has_prob = col_prob in adata_query.obs.columns

    cols_to_pull = [cluster_col, col_main] + ([col_prob] if has_prob else [])
    obs = adata_query.obs[cols_to_pull].copy()
    obs.columns = ["cluster", "label"] + (["prob"] if has_prob else [])

    # Sort cluster IDs: numeric if possible, else lexicographic
    all_ids = obs["cluster"].dropna().unique()
    try:
        cluster_ids = sorted(all_ids, key=lambda x: int(str(x)))
    except (ValueError, TypeError):
        cluster_ids = sorted(all_ids, key=str)

    records = []
    for cid in cluster_ids:
        mask_cluster = obs["cluster"] == cid
        n_total = int(mask_cluster.sum())

        assigned = obs.loc[mask_cluster & obs["label"].notna()]
        n_assigned = len(assigned)
        pct_assigned = round(100.0 * n_assigned / n_total, 2) if n_total else 0.0

        if n_assigned == 0:
            records.append({
                "cluster": cid,
                "n_cells_total": n_total,
                "n_cells_assigned": 0,
                "pct_assigned": 0.0,
                "top_label": pd.NA,
                "top_fraction": pd.NA,
                "mean_prob": pd.NA,
                "margin": pd.NA,
                "second_label": pd.NA,
                "second_fraction": pd.NA,
            })
            continue

        vc = assigned["label"].astype(str).value_counts()
        top_label = vc.index[0]
        top_fraction = round(int(vc.iloc[0]) / n_assigned, 4)

        if len(vc) >= 2:
            second_label   = vc.index[1]
            second_fraction = round(int(vc.iloc[1]) / n_assigned, 4)
            margin          = round(top_fraction - second_fraction, 4)
        else:
            second_label    = pd.NA
            second_fraction = pd.NA
            margin          = pd.NA

        if has_prob:
            top_cells = assigned[assigned["label"].astype(str) == top_label]
            mean_prob = round(float(top_cells["prob"].mean()), 4)
        else:
            mean_prob = pd.NA

        records.append({
            "cluster": cid,
            "n_cells_total": n_total,
            "n_cells_assigned": n_assigned,
            "pct_assigned": pct_assigned,
            "top_label": top_label,
            "top_fraction": top_fraction,
            "mean_prob": mean_prob,
            "margin": margin,
            "second_label": second_label,
            "second_fraction": second_fraction,
        })

    df = pd.DataFrame(records)

    # Ensure object columns are h5ad-serializable in uns:
    # - numeric columns demoted to object by pd.NA → coerce back to float (pd.NA → np.nan)
    # - string columns with pd.NA → fillna("") and cast to str
    _numeric_cols = {"top_fraction", "mean_prob", "margin", "second_fraction"}
    for col in df.columns:
        if df[col].dtype == object:
            if col in _numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].fillna("").astype(str)

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{label_space}_cluster_summary.csv")
        df.to_csv(out_path, index=False)
        _zlog(f"Saved cluster summary → {out_path}")

    return df


# ================================================================
#  4. Build per-cell annotation table
# ================================================================

def build_cell_annotations_table(
    adata_query: ad.AnnData,
    label_space: str,
    *,
    cluster_col: str | None = None,
    time_col: str = "ZMAP_time_id_predicted",
    save_csv: bool = True,
    output_dir: str = "zmap_predict",
) -> pd.DataFrame:
    """
    Build a concise per-cell annotation table from a completed ZMAP run.

    Extracts the annotation-relevant columns from ``adata_query.obs`` into a
    clean, self-contained DataFrame suitable for inspection, CSV export, or
    downstream analysis. Only annotation columns produced by ZMAP are included
    — the full ``obs`` is not copied.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Annotated query dataset.
    label_space : str
        Label namespace used during prediction (matches
        ``adata_query.uns['zmap_labels'][label_space]``).
    cluster_col : str or None, default ``None``
        If provided, include this column (e.g. ``"leiden"``) as the first
        data column so that cells can be linked back to user-defined clusters.
    time_col : str, default ``"ZMAP_time_id_predicted"``
        Column in ``adata_query.obs`` containing predicted developmental time.
        Must match the column written by ``predict_labels_kNN`` (which depends
        on ``time_labels`` and ``label_suffix``).
    save_csv : bool, default ``True``
        Write the table to
        ``{output_dir}/{label_space}_cell_annotations.csv``.

    Returns
    -------
    pd.DataFrame
        One row per cell. ``cell_id`` is the obs index (cell barcode).
        Additional columns are included when present in ``adata_query.obs``:

        - ``{cluster_col}``             — user-defined cluster ID (if provided).
        - ``{label_space}_predicted``   — assigned label (``NA`` if rejected).
        - ``{label_space}_prob``        — kNN vote probability (0–1).
        - ``{label_space}_reject_flag`` — ``True`` if cell failed QC.
        - ``{label_space}_reason``      — which filter triggered rejection.
        - ``{time_col}``                — predicted developmental time (hpf).
    """
    labels_base = f"{label_space}_predicted"
    col_main   = labels_base
    col_prob   = f"{labels_base}_prob"
    col_reject = f"{labels_base}_reject_flag"
    col_reason = f"{labels_base}_reason"

    wanted: list[str] = []
    if cluster_col and cluster_col in adata_query.obs.columns:
        wanted.append(cluster_col)
    for col in [col_main, col_prob, col_reject, col_reason, time_col]:
        if col in adata_query.obs.columns:
            wanted.append(col)

    df = adata_query.obs[wanted].copy()
    df.index.name = "cell_id"
    df = df.reset_index()

    # Ensure object columns are pure str (np.nan in object dtype is not h5ad-serializable in uns)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str)

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{label_space}_cell_annotations.csv")
        df.to_csv(out_path, index=False)
        _zlog(f"Saved cell annotations → {out_path}")

    return df


# ================================================================
#  5. Horizontal histogram (time distribution bar)
# ================================================================

def plot_colorbar_histogram(
    values,
    *,
    bins=100,
    hist_range=None,
    value_min=None,
    value_max=None,
    cmap="Greys",
    vmin=0.0,
    vmax=1.0,
    bar_height=1.0,
    y_min=0,
    y_max=120,
    fig_width=8,
    fig_height=0.6,
    xlabel="Predicted Time (hpf)",
    xlabel_size=15,
    tick_label_size=15,
    title=None,
    title_size=13,
    log=False,
    nan_policy="drop",
    box=True,
    box_lw=1.2,
    box_color="black",
    ax=None,
):
    """
    Plot a colorbar-styled horizontal histogram strip for a distribution of values.

    Renders a single thin bar in which each bin is colored by bin density using
    a colormap, giving a compact "colorbar histogram" suitable for showing
    developmental time distributions alongside UMAP embeddings.

    Used internally by ``plot_embedding_with_ondata_labels`` to draw the
    vertical time strip, but can also be called standalone.

    Parameters
    ----------
    values : array-like
        Numeric values to histogram (e.g. predicted time in hpf). Non-finite
        values are handled according to ``nan_policy``.
    bins : int or array-like, default ``100``
        Number of histogram bins, or explicit bin edges.
    hist_range : tuple of float or None, default ``None``
        ``(min, max)`` range for the histogram. Inferred from data when ``None``.
    value_min, value_max : float or None, default ``None``
        If provided, clip values to ``[value_min, value_max]`` before binning.
        Also sets ``hist_range`` when both are given and ``hist_range`` is ``None``.
    cmap : str, default ``"Greys"``
        Matplotlib colormap name used to color bins by density.
    vmin, vmax : float, default ``0.0`` and ``1.0``
        Colormap normalization range (applied to normalized bin counts).
    bar_height : float, default ``1.0``
        Height of the histogram bar in data units.
    y_min, y_max : float, default ``0`` and ``120``
        Y-axis limits for the plot. ``y_max`` defaults to ``y_min + bar_height``
        when set to ``None``.
    fig_width, fig_height : float, default ``8`` and ``0.6``
        Figure size in inches. Only used when ``ax=None``.
    xlabel : str, default ``"Predicted Time (hpf)"``
        X-axis label.
    xlabel_size, tick_label_size : float, default ``15``
        Font sizes for the axis label and tick labels.
    title : str or None, default ``None``
        Optional title drawn above the strip.
    title_size : float, default ``13``
        Font size for the title.
    log : bool, default ``False``
        If ``True``, apply ``log1p`` to bin counts before coloring.
    nan_policy : str, default ``"drop"``
        How to handle non-finite values. Currently only ``"drop"`` is supported.
    box : bool, default ``True``
        Draw a bounding box around the strip.
    box_lw, box_color : float and str, default ``1.2`` and ``"black"``
        Line width and color for the bounding box.
    ax : matplotlib.axes.Axes or None, default ``None``
        Axes to draw into. If ``None``, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the colorbar histogram strip.
    """
    if y_max is None:
        y_max = y_min + bar_height

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()

    if value_min is not None or value_max is not None:
        lo = -np.inf if value_min is None else value_min
        hi =  np.inf if value_max is None else value_max
        arr = np.clip(arr, lo, hi)
        if hist_range is None and value_min is not None and value_max is not None:
            hist_range = (value_min, value_max)

    finite = np.isfinite(arr)
    if not finite.any():
        if hist_range is None:
            raise ValueError("All values non-finite and no hist_range provided.")
        if isinstance(bins, int):
            counts = np.zeros(bins)
            bin_edges = np.linspace(hist_range[0], hist_range[1], bins+1)
        else:
            bin_edges = np.array(bins)
            counts = np.zeros(len(bin_edges)-1)
    else:
        arr = arr[finite]
        counts, bin_edges = np.histogram(arr, bins=bins, range=hist_range)

    if log:
        counts = np.log1p(counts)

    cmax = counts.max() if counts.size else 0
    norm = counts / cmax if cmax > 0 else counts.copy()
    strip = norm[np.newaxis, :]

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        created = True

    ax.imshow(
        strip, aspect="auto", cmap=cmap,
        extent=[bin_edges[0], bin_edges[-1], y_min, y_max],
        vmin=vmin, vmax=vmax, origin="lower"
    )
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    if title:
        ax.set_title(title, fontsize=title_size)

    if box:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(box_lw)
            s.set_color(box_color)

    if created:
        plt.show()

    return ax


# ================================================================
#  4. Sync color palettes (CellType, Tissue, etc.)
# ================================================================

def sync_zmap_colors(
    adata,
    obs_key='ZMAP_CellType',
    *,
    ref_adata=None,
    ref_obs_key=None,
    unknown_color="#BDBDBD",
):
    """
    Synchronize a categorical color palette between a query and reference AnnData.

    Ensures that ``adata.uns[f"{obs_key}_colors"]`` is populated and aligned
    with the categories in ``adata.obs[obs_key]``. The palette is sourced from
    ``adata.uns`` directly if already present, or copied from ``ref_adata``
    if provided.

    Called automatically by ``plot_embedding_with_ondata_labels``. Call manually
    when you need consistent colors across multiple plots or custom figure code.

    Parameters
    ----------
    adata : anndata.AnnData
        Dataset whose color palette to set or update. Modified in-place.
    obs_key : str, default ``"ZMAP_CellType"``
        Column in ``adata.obs`` whose categories need a synchronized palette.
    ref_adata : anndata.AnnData or None, default ``None``
        Reference dataset from which to copy the palette when ``adata`` does
        not already have one. Looks for ``ref_adata.uns[f"{ref_obs_key}_color_map"]``
        or ``ref_adata.uns[f"{ref_obs_key}_colors"]``.
    ref_obs_key : str or None, default ``None``
        Column in ``ref_adata.obs`` to use as the color source. Defaults to
        ``obs_key`` when ``None``.
    unknown_color : str, default ``"#BDBDBD"``
        Hex color assigned to any category not found in the palette.

    Returns
    -------
    list of str
        Ordered list of hex color strings, one per category in
        ``adata.obs[obs_key].cat.categories``.

    Raises
    ------
    KeyError
        If no palette is found in ``adata.uns`` and ``ref_adata`` is either
        not provided or does not contain a matching palette.
    """
    cmap_key = f"{obs_key}_color_map"

    if cmap_key not in adata.uns:
        if ref_adata is not None:
            src_obs = ref_obs_key or obs_key
            src_cmap_key = f"{src_obs}_color_map"

            if src_cmap_key in ref_adata.uns:
                adata.uns[cmap_key] = dict(ref_adata.uns[src_cmap_key])
            else:
                src_colors = f"{src_obs}_colors"
                if src_colors in ref_adata.uns:
                    cats = ref_adata.obs[src_obs].astype("category").cat.categories
                    cols = ref_adata.uns[src_colors]
                    if len(cols) >= len(cats):
                        adata.uns[cmap_key] = dict(zip(cats, cols))

        if cmap_key not in adata.uns:
            raise KeyError(
                f"No palette found for {obs_key}; provide ref_adata or build manually."
            )

    adata.obs[obs_key] = adata.obs[obs_key].astype("category").cat.remove_unused_categories()

    cats = adata.obs[obs_key].cat.categories
    color_map = adata.uns[cmap_key]
    palette = [color_map.get(c, unknown_color) for c in cats]
    adata.uns[f"{obs_key}_colors"] = palette

    return palette


# ================================================================
#  5. Overlay UMAP: reference background + label and time_id predictions
# ================================================================

def plot_embedding_with_ondata_labels(
    adata_ref,
    adata_test,
    *,
    # ---- data/keys ----
    color_key: str = "ZMAP_Tissue_predicted",
    basis: str = "X_umap",
    ref_basis: str | None = None,
    filter_na: bool = True,

    # ---- palette handling ----
    palette: dict | None = None,
    palette_uns_key: str | None = None,   # defaults to inferred from color_key

    # ---- time strip options ----
    show_time_strip: bool = True,
    time_key: str = "ZMAP_time_id",
    time_strip_width_ratio: float = 0.03,   # fraction of figure width for strip
    time_strip_kwargs: dict | None = None,  # forwarded to plot_colorbar_histogram

    # ---- figure style ----
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 200,

    # ---- point style ----
    ref_size: float = 2,
    ref_alpha: float = 0.3,
    test_size: float = 2,
    test_alpha: float = 1.0,
    show_ref: bool = True,
    cmap: str = "jet",
    frameon: bool = False,
    sort_order: bool = True,
    legend_loc: str = "on data",
    legend_fontsize: float = 5,
    legend_fontweight: str = "normal",

    # ---- label visibility ----
    show_labels: bool = True,

    # ---- label styling ----
    recolor_labels_from_palette: bool = True,
    text_stroke_width: float = 1.0,
    replace_underscores: bool = True,
    linebreak_from: str = "_",
    linebreak_to: str = "\n",

    # ---- adjustText parameters ----
    adjust_expand: tuple[float, float] = (1.2, 1.5),
    arrowprops: dict | None = None,
    min_arrow_len: float = 0,

    # ---- arrow aesthetics ----
    match_arrow_color_to_text: bool = True,
    arrow_alpha: float = 0.8,

    # ---- embedding kwargs passthroughs ----
    ref_kwargs: dict | None = None,
    test_kwargs: dict | None = None,

    # ---- show / save / return behavior ----
    show: bool = False,
    save: bool = True,
    return_ax: bool = False,
    output_dir: str = "zmap_predict",
):
    """
    Plot a query dataset overlaid on the reference embedding, with on-data labels
    and an optional vertical time distribution strip.

    Renders up to two layers: (1) optionally, the full reference embedding
    as a faint grey background for spatial context (controlled by
    ``show_ref``), and (2) the query cells colored by a predicted label
    column. Labels are drawn directly on the embedding using
    ``adjustText`` to minimize overlap. A vertical colorbar histogram of
    predicted developmental time (``ZMAP_time_id``) can optionally be added
    as a strip on the right side of the figure.

    Parameters
    ----------
    adata_ref : anndata.AnnData
        Reference dataset, used only for the background embedding.
    adata_test : anndata.AnnData
        Query dataset with predicted labels to overlay.
    color_key : str, default ``"ZMAP_Tissue_predicted"``
        Column in ``adata_test.obs`` containing the categorical labels to color
        and annotate. Typically a ``_predicted`` column from ``predict_labels_kNN``.
    basis : str, default ``"X_umap"``
        ``obsm`` key used for the 2D embedding coordinates in the query dataset.
    ref_basis : str or None, default ``None``
        ``obsm`` key for the reference background embedding. When ``None``,
        falls back to ``basis``. Useful when the query embedding lives in a
        different key than the reference (e.g. ``basis="X_umap_zmap"`` for
        the query while the reference uses ``"X_umap"``).
    filter_na : bool, default ``True``
        Drop query cells with ``NaN`` in ``color_key`` before plotting.
    palette : dict or None, default ``None``
        Explicit ``{label: color}`` mapping. When ``None``, the palette is
        resolved via ``sync_zmap_colors``.
    palette_uns_key : str or None, default ``None``
        ``uns`` key to look up the palette in ``adata_test``. Inferred from
        ``color_key`` when ``None``.
    show_time_strip : bool, default ``True``
        Draw a vertical colorbar histogram of ``adata_test.obs[time_key]``
        on the right side of the figure.
    time_key : str, default ``"ZMAP_time_id"``
        Column in ``adata_test.obs`` containing predicted developmental time
        values (hours post-fertilization) for the time strip.
    time_strip_width_ratio : float, default ``0.03``
        Width of the time strip as a fraction of the total figure width.
    time_strip_kwargs : dict or None, default ``None``
        Additional keyword arguments forwarded to ``plot_colorbar_histogram``.
    figsize : tuple of float, default ``(6, 6)``
        Figure size in inches ``(width, height)``.
    dpi : int, default ``200``
        Figure resolution.
    ref_size : float, default ``2``
        Scatter point size for reference background cells.
    ref_alpha : float, default ``0.3``
        Opacity of reference background points. Lower values push the
        reference further into the background.
    test_size : float, default ``2``
        Scatter point size for query (projected) cells.
    test_alpha : float, default ``1.0``
        Opacity of query overlay points.
    show_ref : bool, default ``True``
        If ``True``, draw the reference embedding as a faint grey background
        for spatial context. If ``False``, only the query cells are plotted
        (useful when ``do_ingest=False``). When ``False``, ``adata_ref``
        may be ``None``.
    cmap : str, default ``"jet"``
        Colormap used for the reference background scatter.
    legend_loc : str, default ``"on data"``
        Where to place the category legend. ``"on data"`` draws labels directly
        at centroid positions; other values follow matplotlib legend conventions.
        Ignored when ``show_labels=False`` (forced to ``"none"``).
    legend_fontsize, legend_fontweight : float and str, default ``5`` and ``"normal"``
        Font size and weight for on-data legend labels.
    show_labels : bool, default ``True``
        If ``True``, draw on-data text labels at category centroids with
        ``adjustText`` repositioning and optional arrow connectors. If
        ``False``, suppress all text labels and arrows — only the colored
        scatter is shown, which is useful for clean figures or when the
        number of categories is too large for readable labels.
    replace_underscores : bool, default ``True``
        Replace underscores in label strings with line breaks for cleaner
        on-data annotation.
    adjust_expand : tuple of float, default ``(1.2, 1.5)``
        ``(x_expand, y_expand)`` passed to ``adjustText`` for label placement.
    match_arrow_color_to_text : bool, default ``True``
        Color annotation arrows to match their corresponding text label.
    ref_kwargs : dict or None, default ``None``
        Extra keyword arguments forwarded to the reference ``sc.pl.embedding``
        call. Explicit ``ref_alpha`` takes priority over ``alpha`` in this dict.
    test_kwargs : dict or None, default ``None``
        Extra keyword arguments forwarded to the query ``sc.pl.embedding``
        call. Explicit ``test_alpha`` takes priority over ``alpha`` in this dict.
    show : bool, default ``False``
        Call ``plt.show()`` after rendering.
    save : bool, default ``True``
        Save the figure as PNG and PDF to ``output_dir``.
    return_ax : bool, default ``False``
        Return the main ``matplotlib.axes.Axes`` object.

    Returns
    -------
    tuple or None
        ``(fig, ax_umap, ax_strip)`` when ``return_ax=True``, otherwise ``None``.
    """
    # ---- prepare test AnnData (drop NAs on requested key, cast to categorical) ----
    if filter_na:
        mask = ~adata_test.obs[color_key].isna()
        adata_test_plot = adata_test[mask].copy()
    else:
        adata_test_plot = adata_test.copy()

    adata_test_plot.obs[color_key] = (
        adata_test_plot.obs[color_key].astype("category").cat.remove_unused_categories()
    )

    # ---- sync palettes using base obs key (before _predicted) ----
    base_obs = color_key.replace("_predicted", "")

    # Only attempt sync if there's no _color_map dict already available
    # (annotate_with_zmap populates _color_map from reference colormaps)
    cmap_dict_key = f"{base_obs}_color_map"
    if cmap_dict_key not in adata_test_plot.uns and adata_ref is not None:
        try:
            sync_zmap_colors(adata_ref, obs_key=base_obs)
            sync_zmap_colors(adata_test_plot, obs_key=color_key, ref_adata=adata_ref, ref_obs_key=base_obs)
        except Exception as e:
            warnings.warn(
                f"[ZMAP] sync_zmap_colors failed for '{color_key}': {e}. "
                "Falling back to _color_map dict or positional palette.",
                stacklevel=2,
            )

    # ---- palette construction ----
    if palette is None:
        cats = adata_test_plot.obs[color_key].cat.categories

        # Preferred: use _color_map dict (order-independent, set by annotate_with_zmap)
        cmap_dict_key = f"{base_obs}_color_map"
        if cmap_dict_key in adata_test_plot.uns:
            color_map = adata_test_plot.uns[cmap_dict_key]
            palette = {c: color_map.get(c, "#BDBDBD") for c in cats}
        else:
            # Legacy fallback: positional _colors array
            if palette_uns_key is None:
                palette_uns_key = f"{base_obs}_colors"
            if palette_uns_key not in adata_test_plot.uns:
                raise KeyError(
                    f"Palette not provided and neither '{cmap_dict_key}' nor "
                    f"'{palette_uns_key}' found in adata.uns. "
                    f"Provide `palette` or ensure a color map exists."
                )
            colors = adata_test_plot.uns[palette_uns_key]
            palette = dict(zip(cats, colors))
            warnings.warn(
                f"[ZMAP] Using positional _colors array for '{base_obs}'; "
                "results may be wrong if query categories differ from reference order. "
                "Consider storing a _color_map dict.",
                stacklevel=2,
            )

    # ---- defaults for arrowprops ----
    if arrowprops is None:
        arrowprops = dict(arrowstyle='-', color='k', lw=0.25)

    ref_kwargs = {} if ref_kwargs is None else dict(ref_kwargs)
    test_kwargs = {} if test_kwargs is None else dict(test_kwargs)
    time_strip_kwargs = {} if time_strip_kwargs is None else dict(time_strip_kwargs)

    # ---- Inject alpha into kwargs (explicit params take priority) ----
    ref_kwargs.setdefault("alpha", ref_alpha)
    test_kwargs.setdefault("alpha", test_alpha)

    # ---- Resolve legend_loc: suppress labels when show_labels=False ----
    legend_loc_use = legend_loc if show_labels else "none"

    # ---- Check if we actually have a time vector ----
    has_time = (
        show_time_strip and
        (time_key in adata_test.obs.columns) and
        pd.to_numeric(adata_test.obs[time_key], errors="coerce").notna().any()
    )

    fig = None
    ax_umap = None
    ax_strip = None

    # ---- Resolve ref_basis: defaults to basis if not provided ----
    ref_basis_use = ref_basis if ref_basis is not None else basis

    # ---- figure + axes layout ----
    with plt.rc_context({'figure.figsize': figsize, 'figure.dpi': dpi}):
        if has_time:
            # two-column layout: [UMAP | time strip]
            fig = plt.figure()
            gs = fig.add_gridspec(
                1,
                2,
                width_ratios=[1.0 - time_strip_width_ratio, time_strip_width_ratio],
                wspace=0.05,
            )
            ax_umap  = fig.add_subplot(gs[0, 0])   # UMAP on the left
            ax_strip = fig.add_subplot(gs[0, 1])   # colorbar on the right
        else:
            fig, ax_umap = plt.subplots()
            ax_strip = None

        # ---- reference embedding in UMAP axis (optional) ----
        if show_ref:
            ax = sc.pl.embedding(
                adata_ref,
                basis=ref_basis_use,
                show=False,
                s=ref_size,
                ax=ax_umap,
                **ref_kwargs,
            )
        else:
            ax = ax_umap

        # ---- query overlay in UMAP axis ----
        ax = sc.pl.embedding(
            adata_test_plot,
            color=color_key,
            frameon=frameon,
            legend_fontsize=legend_fontsize,
            cmap=cmap,
            basis=basis,
            sort_order=sort_order,
            size=test_size,
            legend_loc=legend_loc_use,
            legend_fontweight=legend_fontweight,
            ax=ax,
            show=False,
            title=color_key + " (Predicted)",
            palette=palette,
            **test_kwargs,
        )

        # --- on-data label styling (only when show_labels=True) ---
        if show_labels:
            texts = [t for t in ax.texts]

            # --- recolor existing on-data text labels by palette ---
            if recolor_labels_from_palette:
                for t in texts:
                    label = t.get_text()
                    if label in palette:
                        t.set_color(palette[label])

            # --- add white border to text labels ---
            for t in texts:
                t.set_path_effects([pe.withStroke(linewidth=text_stroke_width, foreground='white')])

            # --- replace underscores with line breaks ---
            if replace_underscores:
                for t in texts:
                    t.set_text(t.get_text().replace(linebreak_from, linebreak_to))

            # --- adjust label positions with slim arrows ---
            if len(texts) > 0:
                adjust_text(
                    texts,
                    ax=ax,
                    expand=adjust_expand,
                    arrowprops=arrowprops,
                    min_arrow_len=min_arrow_len,
                )

            # --- match arrow color to label color ---
            if match_arrow_color_to_text and len(texts) > 0:
                arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
                for t, a in zip(texts, arrows[-len(texts):]):
                    a.set_color(t.get_color())
                    a.set_alpha(arrow_alpha)

        # ---- time distribution strip on the right (rotated) ----
        if has_time and ax_strip is not None:
            # 1) Draw horizontal strip into a temporary axis
            fig_tmp, ax_tmp = plt.subplots(figsize=(2, 2))
            plot_colorbar_histogram(
                adata_test.obs[time_key],
                ax=ax_tmp,
                xlabel="",          # suppress label in temp axis
                fig_width=2,
                fig_height=0.4,
                **time_strip_kwargs,
            )

            # Grab the image that was drawn
            im = ax_tmp.get_images()[0]
            arr = im.get_array()   # shape (1, N) — horizontal strip
            xmin, xmax, ymin, ymax = im.get_extent()
            # original: extent = [bin_edges[0], bin_edges[-1], y_min, y_max]
            plt.close(fig_tmp)

            # 2) Transpose to make it vertical without reversing bin order
            arr_vert = arr.T  # (N, 1)

            # 3) Plot vertical strip into ax_strip
            # x in [0, 1] (thin bar), y in [xmin, xmax] (time axis)
            ax_strip.imshow(
                arr_vert,
                aspect="auto",
                cmap=im.get_cmap(),
                origin="lower",
                interpolation="nearest",
                extent=[0.0, 1.0, xmin, xmax],
            )

            # Cosmetics: side colorbar feel
            ax_strip.set_xticks([])
            ax_strip.set_xlim(0.0, 1.0)
            ax_strip.set_ylabel("Predicted Time (hpf)")

            # Nudge the colorbar rightwards a bit
            pos = ax_strip.get_position()
            ax_strip.set_position([pos.x0 + 0.1, pos.y0, pos.width, pos.height])

            # Shrink colorbar height by 50% and vertically center
            pos = ax_strip.get_position()
            new_height = pos.height * 0.50
            new_bottom = pos.y0 + (pos.height - new_height) / 2
            ax_strip.set_position([pos.x0, new_bottom, pos.width, new_height])

            # Reinforce border box
            for s in ax_strip.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.8)
                s.set_color("black")

        # ---- SAVE FIGURE ----
        if save:
            os.makedirs(output_dir, exist_ok=True)

            safe_name = color_key.replace("/", "_").replace(" ", "_")
            png_path = os.path.join(output_dir, f"{safe_name}.png")
            pdf_path = os.path.join(output_dir, f"{safe_name}.pdf")

            fig.savefig(png_path, dpi=dpi*3, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            _zlog(f"Saved figure:\n - {png_path}\n - {pdf_path}")

        # ---- SHOW FIGURE ----
        if show:
            plt.show()
        else:
            # Close figure to prevent Jupyter/Colab inline backend from
            # auto-displaying it at cell completion
            plt.close(fig)

    # ---- RETURN AXES (OPTIONAL) ----
    if return_ax:
        return fig, ax_umap, ax_strip

    return None


# ================================================================
#  6. Overlap matrix & plot
# ================================================================

def map_query_labels(
    adata_query,
    obs_A: str,
    obs_B: str,
    *,
    normalize="row",              # "row" | "column" | None | True | False
    title=None,
    reorder_columns=True,
    reorder_rows=True,
    cmap=plt.cm.Blues,
    overlay_values=False,
    vmin=None,
    vmax=None,
    show_plot=True,
    return_df=False,              # return mapping_df
    figsize=8,
    save_plots=True,              # save PNG + PDF
    save_mapping=True,            # save mapping_df to CSV
    file_prefix: str | None = None,  # optional prefix for output filenames; defaults to obs_A
    output_dir: str = "zmap_predict",
):
    """
    Compute and visualize the overlap between two label columns in a query AnnData.

    Builds a contingency matrix comparing two categorical ``obs`` columns
    (e.g. ZMAP predicted labels vs. Leiden clusters), applies optional
    row- or column-wise normalization, and plots the result as a heatmap.
    Also computes a per-group best-match mapping table.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Annotated query dataset containing both label columns.
    obs_A : str
        Column in ``adata_query.obs`` used as the reference labeling
        (appears as columns in the overlap matrix).
    obs_B : str
        Column in ``adata_query.obs`` used as the query labeling
        (appears as rows in the overlap matrix).
    normalize : str or None, default ``"row"``
        Normalization applied to the raw overlap counts before plotting.
        One of:

        - ``"row"``    — each row sums to 1 (fraction of obs_B in each obs_A).
        - ``"column"`` — each column sums to 1 (fraction of obs_A in each obs_B).
        - ``None``     — plot raw cell counts.

        ``True`` is treated as ``"row"`` and ``False`` as ``None`` for
        backward compatibility.
    title : str or None, default ``None``
        Plot title. Auto-generated from ``obs_A`` and ``obs_B`` when ``None``.
    reorder_columns : bool, default ``True``
        Sort columns by the position of their best-matching row.
    reorder_rows : bool, default ``True``
        Sort rows by the position of their best-matching column.
    cmap : matplotlib colormap, default ``plt.cm.Blues``
        Colormap for the heatmap.
    overlay_values : bool, default ``False``
        Overlay numeric values in each heatmap cell.
    vmin, vmax : float or None, default ``None``
        Colormap normalization limits.
    show_plot : bool, default ``True``
        Display the plot immediately.
    return_df : bool, default ``False``
        Return the best-match mapping table as a ``pd.DataFrame``.
    figsize : float, default ``8``
        Figure size (passed as both width and height in inches).
    save_plots : bool, default ``True``
        Save PNG and PDF of the heatmap to ``./zmap/predict/``.
    save_mapping : bool, default ``True``
        Save the best-match mapping table as a CSV to ``./zmap/predict/``.

    Returns
    -------
    pd.DataFrame or None
        When ``return_df=True``, a per-group best-match table mapping each
        obs_B label to its most-overlapping obs_A label. ``None`` otherwise.
    """

    # --------------------------------------------------------------------------
    # 1. Normalize argument handling
    # --------------------------------------------------------------------------
    if isinstance(normalize, bool):
        normalize = "row" if normalize else None
    valid_norm = {"row", "column", None}
    if normalize not in valid_norm:
        raise ValueError(f"normalize must be one of {valid_norm}, got {normalize!r}")

    # --------------------------------------------------------------------------
    # 2. Fetch columns and build raw overlap table
    # --------------------------------------------------------------------------
    labels_A = adata_query.obs[obs_A]
    labels_B = adata_query.obs[obs_B]

    labels_A = pd.Series(labels_A)
    labels_B = pd.Series(labels_B)

    mask = labels_A.notna() & labels_B.notna()
    labels_A = labels_A[mask]
    labels_B = labels_B[mask]

    overlap_table = pd.crosstab(labels_B, labels_A)
    if overlap_table.empty:
        raise ValueError("Overlap matrix is empty (no overlap or only NaNs).")

    # --------------------------------------------------------------------------
    # 3. Normalization
    # --------------------------------------------------------------------------
    if normalize == "row":
        denom = overlap_table.sum(axis=1).replace(0, np.nan)
        plot_table = overlap_table.div(denom, axis=0).fillna(0)
        colorbar_label = "Fraction overlap (row-normalized)"
        if title is None:
            title = f"{obs_B} → {obs_A} (row-normalized)"

    elif normalize == "column":
        denom = overlap_table.sum(axis=0).replace(0, np.nan)
        plot_table = overlap_table.div(denom, axis=1).fillna(0)
        colorbar_label = "Fraction overlap (column-normalized)"
        if title is None:
            title = f"{obs_B} → {obs_A} (column-normalized)"

    else:
        plot_table = overlap_table.astype(float)
        colorbar_label = "# overlaps"
        if title is None:
            title = f"{obs_B} → {obs_A} (counts)"

    # --------------------------------------------------------------------------
    # 4. Reorder rows/columns
    # --------------------------------------------------------------------------
    arr = plot_table.to_numpy()
    row_labels = plot_table.index.to_numpy()
    col_labels = plot_table.columns.to_numpy()

    if reorder_columns and arr.size > 0:
        idx = np.argsort(np.argmax(arr, axis=0))
        arr = arr[:, idx]
        col_labels = col_labels[idx]

    if reorder_rows and arr.size > 0:
        idx = np.argsort(np.argmax(arr, axis=1))
        arr = arr[idx, :]
        row_labels = row_labels[idx]

    plot_table = pd.DataFrame(arr, index=row_labels, columns=col_labels)

    # --------------------------------------------------------------------------
    # 5. Plotting
    # --------------------------------------------------------------------------
    fig = None
    if show_plot or save_plots:
        plt.rcParams["axes.grid"] = False
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        im = ax.imshow(plot_table.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_aspect("equal")
        ax.set_xticks(np.arange(plot_table.shape[1]))
        ax.set_yticks(np.arange(plot_table.shape[0]))
        ax.set_xticklabels(plot_table.columns, rotation=90)
        ax.set_yticklabels(plot_table.index)
        ax.set_title(title)
        ax.set_xlabel(obs_A)
        ax.set_ylabel(obs_B)

        cb = fig.colorbar(im, ax=ax, shrink=0.5)
        cb.ax.set_ylabel(colorbar_label)

        if overlay_values:
            vals = plot_table.to_numpy()
            thresh = vals.max() / 2 if vals.size else 0
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    val = vals[i, j]
                    txt = f"{val:.2f}" if normalize in {"row", "column"} else f"{int(val)}"
                    ax.text(
                        j, i, txt,
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=8,
                    )

        plt.tight_layout()

    # --------------------------------------------------------------------------
    # 6. Compute mapping_df (always computed)
    # --------------------------------------------------------------------------
    raw_reordered = overlap_table.loc[row_labels, col_labels]
    top_match = raw_reordered.idxmax(axis=1)
    mapping_df = pd.DataFrame({"top_match": top_match})

    # Pretty sorting if index numeric
    idx_str = mapping_df.index.astype(str)
    if all(s.isdigit() for s in idx_str):
        mapping_df.index = mapping_df.index.astype(int)
        mapping_df = mapping_df.sort_index()

    # --------------------------------------------------------------------------
    # 7. Saving (always applies when save_mapping=True)
    # --------------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    prefix = file_prefix or obs_A

    if save_plots and fig is not None:
        base = f"{prefix}_{obs_B}_overlap"
        fig.savefig(os.path.join(output_dir, f"{base}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"{base}.pdf"), bbox_inches="tight")
        _zlog(f"Saved overlap figure → {output_dir}/{base}.png")

    if save_mapping and mapping_df is not None:
        out_csv = os.path.join(output_dir, f"{prefix}_{obs_B}_top_label.csv")
        mapping_df.to_csv(out_csv)
        _zlog(f"Saved top-label mapping → {out_csv}")

    # Show or close the figure
    if fig is not None:
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    # --------------------------------------------------------------------------
    # 8. Return mapping_df or None
    # --------------------------------------------------------------------------
    return mapping_df if return_df else None


# ================================================================
#  7. Full pipeline wrapper
# ================================================================

def annotate_with_zmap(
    adata_query: ad.AnnData,
    *,
    # --- where raw counts live ---
    query_raw_counts_source: str,                     # "X" or a layer name

    # --- reference handling ---
    adata_ref: ad.AnnData | None = None,
    ref_kind: str = "symphony",
    ref_label_col: str = "ZMAP_CellType",
    label_space: str | None = None,
    query_truth_col: str | None = None,

    # --- query label aggregation ---
    query_label_col: str | None = None,   # user-defined labels/clusters, e.g. "leiden"
    cluster_col: str | None = None,       # deprecated alias for query_label_col

    # --- pipeline toggles ---
    do_preprocess: bool = True,
    do_map_embedding: bool = True,
    do_ingest: bool = True,
    tissue_aware: bool = False,       # use tissue-aware kNN transfer
    evaluate: bool = False,           # compute accuracy metrics against query_truth_col
    n_neighbors: int = 25,            # number of neighbors for kNN voting
    marker_validation: bool = True,   # validate predicted labels against ZMAP marker ledger

    # --- kwargs passthroughs to lower-level steps ---
    preprocess_kwargs: Mapping[str, Any] | None = None,
    predict_kwargs: Mapping[str, Any] | None = None,

    # --- output controls ---
    verbosity: int = 2,               # 0=silent, 1=progress, 2=summary+UMAP+QC, 3=full
    debug: bool = False,              # if True, re-raise exceptions instead of catching
    print_summary: bool | None = None,   # deprecated → use verbosity
    show_plots: bool | None = None,      # deprecated → use verbosity
    save_outputs: bool = True,            # save CSVs and PNGs to output_dir
    output_dir: str = "zmap_predict",     # base directory; files go to output_dir/{label_space}/
) -> ad.AnnData:
    """
    End-to-end ZMAP annotation pipeline: preprocess → embed → transfer labels → plot.

    This is the primary entry point for annotating a new single-cell dataset
    with ZMAP reference labels. It chains the following steps:

    1. **Preprocess** — normalize raw counts to TPM + log1p (``preprocess_adata_query``).
    2. **Embed** — map the query into the ZMAP Symphony PCA embedding and ingest
       into the reference UMAP (requires ``symphonypy``).
    3. **Label transfer** — kNN voting to assign cell-type, tissue, and time labels
       (``predict_labels_kNN``; optional tissue-aware mode via
       ``predict_labels_tissue_kNN``).
    4. **Summarize** — store a simplified run summary in
       ``adata_query.uns['zmap_labels'][<space>]['Run Summary Simple']``.
    5. **Plot** — overlay query cells on the reference UMAP with on-data labels
       (``plot_embedding_with_ondata_labels``).
    6. **Map labels** *(optional)* — cross-tabulate ZMAP labels against an existing
       query labeling (e.g. Leiden clusters) via ``map_query_labels``.

    All run parameters are stored in
    ``adata_query.uns['zmap_labels'][label_space]['_run_config']`` so that
    on-demand accessors (``plot_qc``, ``plot_embedding``, ``plot_time``,
    ``plot_overlap_matrix``, ``show_summary``) can reproduce pipeline outputs
    with just ``adata_query`` — no extra arguments needed.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Query dataset to annotate. Modified in-place.
    query_raw_counts_source : str
        Where raw integer counts are stored in ``adata_query``. Pass ``"X"``
        to use ``adata_query.X``, or a layer name (e.g. ``"counts"``) to use
        ``adata_query.layers[query_raw_counts_source]``. Required — no default.
    adata_ref : anndata.AnnData or None, default ``None``
        Pre-loaded ZMAP reference object. When ``None``, the reference is loaded
        automatically using ``load_zmap_h5ad(kind=ref_kind)``.
    ref_kind : str, default ``"symphony"``
        Which reference preset to load when ``adata_ref=None``. Passed to
        ``load_zmap_h5ad``. Use ``"symphony"`` for label transfer.
    ref_label_col : str, default ``"ZMAP_CellType"``
        Column in the reference ``obs`` whose labels are transferred to the query.
        Also controls which UMAP overlay plot is generated.
    label_space : str or None, default ``None``
        Namespace for output columns and ``uns`` keys. Defaults to ``ref_label_col``.
    query_truth_col : str or None, default ``None``
        Ground-truth label column in ``adata_query.obs``, used for evaluation
        metrics when ``evaluate=True``.
    query_label_col : str or None, default ``None``
        Column in ``adata_query.obs`` containing user-defined cluster or label
        IDs (e.g. ``"leiden"``). When provided, enables cluster-level consensus
        aggregation and the label-overlap matrix. Recommended for most workflows.
    cluster_col : str or None, default ``None``
        Deprecated alias for ``query_label_col``.
    do_preprocess : bool, default ``True``
        Run TPM normalization + log1p on the query before mapping.
        Set to ``False`` if ``adata_query.X`` is already log-normalized.
    do_map_embedding : bool, default ``True``
        Run Symphony embedding mapping. Requires ``symphonypy``. Set to ``False``
        if the query already has a ``X_pca_harmony`` embedding.
    do_ingest : bool, default ``True``
        Ingest the query into the reference UMAP after Symphony mapping.
        Only applies when ``do_map_embedding=True``.
    tissue_aware : bool, default ``False``
        Use tissue-aware kNN transfer (``predict_labels_tissue_kNN``).
        Equivalent to ``predict_kwargs={"use_tissue_aware_knn": True,
        "auto_pseudo_tissue": True}``. When ``True``, any additional
        tissue-aware options can still be passed via ``predict_kwargs``.
    evaluate : bool, default ``False``
        Compute accuracy and evaluation metrics against ``query_truth_col``.
        Requires ``query_truth_col`` to be set. Equivalent to
        ``predict_kwargs={"evaluate": True, "plot_eval_curves": True}``.
    n_neighbors : int, default ``25``
        Number of nearest neighbors for kNN label voting. With Gaussian
        distance weighting (the default), 25 is robust — distant neighbors
        are downweighted automatically, so the effective neighborhood adapts
        to local density.
    marker_validation : bool, default ``True``
        Validate predicted labels by comparing DE markers against the ZMAP
        consensus marker ledger. Discovers the top 20 DE genes per predicted
        group and measures overlap with the top 100 reference markers.
        Results are stored in
        ``adata_query.uns['zmap_labels'][label_space]['Marker Validation']``.
    preprocess_kwargs : dict or None, default ``None``
        Extra keyword arguments forwarded to ``preprocess_adata_query``
        (e.g. ``{"strict_counts": True}``).
    predict_kwargs : dict or None, default ``None``
        Extra keyword arguments forwarded to ``predict_labels_kNN``.
        For common options, prefer the top-level ``tissue_aware`` and
        ``evaluate`` parameters instead of passing dicts manually.
    verbosity : int, default ``2``
        Controls how much output is printed and displayed inline:

        - ``0`` — silent (no print, no inline plots).
        - ``1`` — progress lines only (``[ZMAP] Step complete (Xs)``).
        - ``2`` — compact summary + UMAP overlay + combined QC figure.
        - ``3`` — full display: all tables via ``display()``, all plots
          including heatmap.
    debug : bool, default ``False``
        If ``True``, re-raise exceptions from plotting and aggregation steps
        instead of catching them. Useful for development and troubleshooting.
    print_summary : bool or None, default ``None``
        Deprecated. Use ``verbosity`` instead. When explicitly set,
        ``False`` caps verbosity at 0.
    show_plots : bool or None, default ``None``
        Deprecated. Use ``verbosity`` instead. When explicitly set,
        ``False`` caps verbosity at 1.
    save_outputs : bool, default ``True``
        Save cell annotations CSV, cluster summary CSV, and all figures
        to ``{output_dir}/{label_space}/``.

    Returns
    -------
    anndata.AnnData
        The annotated query dataset (same object, modified in-place). Key
        additions to ``adata_query``:

        - ``.obs[f"{label_space}_predicted"]``                       — transferred cell labels.
        - ``.obs[f"{label_space}_prob"]``                            — label confidence (0–1).
        - ``.obs["ZMAP_time_id_predicted"]``                         — predicted time (hpf).
        - ``.obsm["X_umap_zmap"]``                                   — projected UMAP coordinates (if ingested).
        - ``.uns['zmap_labels']['_last_space']``                     — most recent label_space.
        - ``.uns['zmap_labels'][label_space]['_run_config']``        — stored run parameters
          for zero-arg on-demand plot accessors.
        - ``.uns['zmap_labels'][label_space]['Run Summary Simple']`` — key/value run summary.
        - ``.uns['zmap_labels'][label_space]['Cell Annotations']``   — per-cell table.
        - ``.uns['zmap_labels'][label_space]['Cluster Summary']``    — cluster consensus table
          (only when ``query_label_col`` is provided).
        - ``.uns['zmap_labels'][label_space]['Label Mapping']``      — label overlap matrix
          (only when ``query_label_col`` is provided).
        - ``.uns['zmap_labels'][label_space]['Marker Validation']`` — DE marker overlap with
          ZMAP reference ledger (only when ``marker_validation=True``).

    Examples
    --------
    Minimal usage:

    >>> adata = zmap.predict.annotate_with_zmap(
    ...     adata_query,
    ...     query_raw_counts_source="counts",
    ...     query_label_col="leiden",
    ... )

    Tissue-aware mode:

    >>> adata = zmap.predict.annotate_with_zmap(
    ...     adata_query,
    ...     query_raw_counts_source="counts",
    ...     tissue_aware=True,
    ... )

    Evaluation mode with ground-truth labels:

    >>> adata = zmap.predict.annotate_with_zmap(
    ...     adata_query,
    ...     query_raw_counts_source="counts",
    ...     query_truth_col="manual_annotation",
    ...     evaluate=True,
    ... )

    Re-plot any output with zero arguments:

    >>> zmap.predict.plot_qc(adata)
    >>> zmap.predict.plot_embedding(adata)
    >>> zmap.predict.plot_overlap_matrix(adata)
    >>> zmap.predict.show_summary(adata)
    """

    # Suppress UMAP "n_jobs overridden" warnings
    warnings.filterwarnings(
        "ignore",
        message=".*overridden to 1 by setting random_state.*"
    )

    # Start pipeline clock for _zlog timestamps
    global _T_PIPELINE_START
    _T_PIPELINE_START = time.time()

    # ------------------------------------------------------------------
    # Resolve verbosity (handle deprecated print_summary / show_plots)
    # ------------------------------------------------------------------
    v = int(verbosity)
    if print_summary is not None or show_plots is not None:
        warnings.warn(
            "print_summary and show_plots are deprecated. Use verbosity=0..3 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if print_summary is not None and not print_summary:
            v = min(v, 0)
        if show_plots is not None and not show_plots:
            v = min(v, 1)

    # ------------------------------------------------------------------
    # 0. Resolve query_label_col (handle deprecated cluster_col alias)
    # ------------------------------------------------------------------
    if cluster_col is not None and query_label_col is None:
        warnings.warn(
            "cluster_col is deprecated and will be removed in a future version. "
            "Use query_label_col instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        query_label_col = cluster_col

    if query_label_col is None:
        if v >= 1:
            _zlog(
                "Note: no query_label_col supplied. Cluster-level consensus aggregation "
                "will be skipped. Pass query_label_col='leiden' (or similar) for a complete summary."
            )
    elif query_label_col not in adata_query.obs.columns:
        if v >= 1:
            _zlog(
                f"Warning: query_label_col '{query_label_col}' not found in adata_query.obs. "
                "Cluster aggregation will be skipped."
            )
        query_label_col = None

    # ------------------------------------------------------------------
    # 1. Load reference if needed
    # ------------------------------------------------------------------
    if adata_ref is None:
        if v >= 1:
            _zlog(f"Loading reference ({ref_kind})...")
        from zmap.reference import load_zmap_h5ad
        adata_ref = load_zmap_h5ad(kind=ref_kind)
        if v >= 1:
            _zlog("Reference loaded.")

    # Effective label namespace
    space = label_space or ref_label_col

    # ------------------------------------------------------------------
    # Output directory: zmap_predict/{label_space}/
    # ------------------------------------------------------------------
    space_dir = os.path.join(output_dir, space)
    if save_outputs:
        os.makedirs(space_dir, exist_ok=True)
        existing = [f for f in os.listdir(space_dir) if f.startswith(space)]
        if existing and v >= 1:
            _zlog(f"Note: overwriting {len(existing)} existing file(s) in {space_dir}/")

    # ------------------------------------------------------------------
    # 2. Preprocess query (TPM + log1p)
    # ------------------------------------------------------------------
    if do_preprocess:
        if v >= 1:
            _zlog("Preprocessing query — TPM normalization + log1p ...")
        pp_kwargs = dict(preprocess_kwargs or {})
        preprocess_adata_query(
            adata_query,
            counts_source=query_raw_counts_source,
            **pp_kwargs,
        )
        if v >= 1:
            _zlog("Preprocessing complete.")

    # ------------------------------------------------------------------
    # 3. Symphony mapping / UMAP ingest
    # ------------------------------------------------------------------
    if do_map_embedding:
        if v >= 1:
            _zlog("Mapping query to ZMAP Symphony embedding...")

        try:
            import symphonypy as sp
        except ImportError as e:
            raise ImportError(
                "Symphony (`import symphonypy as sp`) is required for mapping. "
                "Install symphonypy or disable do_map_embedding."
            ) from e

        t0_map = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sp.tl.map_embedding(adata_query, adata_ref)
        if v >= 1:
            _zlog(f"Mapping complete ({time.time() - t0_map:.0f}s).")

        if do_ingest:
            if v >= 1:
                _zlog("Ingesting query into reference UMAP...")

            # Stash pre-existing X_umap so symphony doesn't clobber it
            _had_existing_umap = "X_umap" in adata_query.obsm
            if _had_existing_umap:
                _existing_umap = adata_query.obsm["X_umap"].copy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                sp.tl.ingest(adata_query=adata_query, adata_ref=adata_ref)

            # Move ingested coords to X_umap_zmap (non-destructive)
            adata_query.obsm["X_umap_zmap"] = adata_query.obsm["X_umap"].copy()

            # Restore original X_umap if one existed, otherwise remove
            if _had_existing_umap:
                adata_query.obsm["X_umap"] = _existing_umap
                if v >= 1:
                    _zlog("Restored pre-existing X_umap; ingested coords in X_umap_zmap.")
            else:
                del adata_query.obsm["X_umap"]
                if v >= 1:
                    _zlog("Ingested coords stored in X_umap_zmap.")

            if v >= 1:
                _zlog("Ingestion complete.")

    # ------------------------------------------------------------------
    # 4. kNN label transfer
    # ------------------------------------------------------------------
    if v >= 1:
        _zlog("Running kNN-based label transfer...")
    pk = dict(predict_kwargs or {})
    pk.setdefault("ref_basis", "X_pca_harmony")
    pk.setdefault("query_basis", "X_pca_harmony")
    pk.setdefault("metric", "cosine")
    pk.setdefault("n_neighbors", n_neighbors)
    pk.setdefault("label_suffix", "predicted")   # ensures obs columns are always {space}_predicted
    pk.setdefault("output_dir", space_dir)
    # QC plots: always generate+save if save_outputs; display inline at v>=2
    pk.setdefault("show_qc_plots", v >= 2)
    pk.setdefault("save_mapping_qc", save_outputs)
    # Top-level shortcuts → inject into predict_kwargs
    if tissue_aware:
        pk.setdefault("use_tissue_aware_knn", True)
        pk.setdefault("auto_pseudo_tissue", True)
    if evaluate:
        if not query_truth_col:
            raise ValueError(
                "evaluate=True requires query_truth_col to be set."
            )
        pk.setdefault("evaluate", True)
        pk.setdefault("plot_eval_curves", True)
    use_tissue_aware_knn = bool(
        pk.pop("use_tissue_aware_knn", False) or pk.pop("use_tissue_aware", False)
    )

    t0 = time.time()
    if use_tissue_aware_knn:
        if v >= 1:
            _zlog("Using tissue-aware kNN transfer...")
        predict_labels_tissue_kNN(
            adata_query,
            adata_ref,
            ref_label_col=ref_label_col,
            label_space=space,
            query_truth_col=query_truth_col,
            **pk,
        )
    else:
        predict_labels_kNN(
            adata_query,
            adata_ref,
            ref_label_col=ref_label_col,
            label_space=space,
            query_truth_col=query_truth_col,
            **pk,
        )
    if v >= 1:
        _zlog(f"Label transfer finished ({time.time() - t0:.0f}s).")

    # ------------------------------------------------------------------
    # 4b. Resolve actual time column name (matches predict_labels_kNN logic)
    # ------------------------------------------------------------------
    _label_suffix = pk.get("label_suffix", "predicted") or ""
    _time_labels = pk.get("time_labels", "time_id")
    if _label_suffix:
        time_col_actual = f"ZMAP_{_time_labels}_{_label_suffix}"
    else:
        time_col_actual = f"ZMAP_{_time_labels}"

    # ------------------------------------------------------------------
    # 4c. Store run config for zero-arg on-demand accessors
    # ------------------------------------------------------------------
    adata_query.uns.setdefault("zmap_labels", {})
    adata_query.uns["zmap_labels"]["_last_space"] = space
    adata_query.uns["zmap_labels"].setdefault(space, {})
    # Determine embedding basis and whether to show reference background
    if do_ingest:
        _plot_basis = "X_umap_zmap"
        _show_ref = True
    else:
        # No ingest → query lives in its own UMAP space
        _plot_basis = "X_umap" if "X_umap" in adata_query.obsm else None
        _show_ref = False

    adata_query.uns["zmap_labels"][space]["_run_config"] = {
        "label_space": space,
        "ref_label_col": ref_label_col,
        "query_label_col": query_label_col,
        "time_col": time_col_actual,
        "output_dir": space_dir,
        "basis": _plot_basis,
        "show_ref": _show_ref,
    }
    # Store reference UMAP for on-demand plot_embedding (only when ingested)
    if do_ingest and "X_umap" in adata_ref.obsm:
        adata_query.uns["zmap_labels"][space]["_ref_umap"] = (
            np.asarray(adata_ref.obsm["X_umap"]).copy()
        )

    # ------------------------------------------------------------------
    # 5. Run summary (key/value metadata table)
    # ------------------------------------------------------------------
    df_summary = summarize_knn_run(adata_query, space)

    adata_query.uns.setdefault("zmap_labels", {}).setdefault(space, {})
    adata_query.uns["zmap_labels"][space]["Run Summary Simple"] = df_summary

    if v >= 3:
        _zlog("── Run Summary ──────────────────────────────────────")
        _display_df(df_summary)

    # ------------------------------------------------------------------
    # 6. Per-cell annotation table
    # ------------------------------------------------------------------
    if v >= 1:
        _zlog("Building per-cell annotation table...")
    df_cells = build_cell_annotations_table(
        adata_query,
        space,
        cluster_col=query_label_col,
        time_col=time_col_actual,
        save_csv=save_outputs,
        output_dir=space_dir,
    )
    adata_query.uns["zmap_labels"][space]["Cell Annotations"] = df_cells

    if v >= 3:
        n_cells = len(df_cells)
        _zlog(f"\n── Cell Annotations ({n_cells:,} cells) ──────────────────")
        _display_df(df_cells, max_rows=10)

    # ------------------------------------------------------------------
    # 7. Cluster-level consensus aggregation
    # ------------------------------------------------------------------
    if query_label_col is not None:
        if v >= 1:
            _zlog(f"Aggregating cell annotations by cluster ('{query_label_col}')...")
        try:
            df_clusters = aggregate_by_cluster(
                adata_query,
                cluster_col=query_label_col,
                label_space=space,
                save_csv=save_outputs,
                output_dir=space_dir,
            )
            adata_query.uns["zmap_labels"][space]["Cluster Summary"] = df_clusters

            if v >= 3:
                n_clusters = len(df_clusters)
                _zlog(f"\n── Cluster Summary ({n_clusters} clusters) ─────────────────")
                _display_df(df_clusters, max_rows=20)

            if v >= 1:
                _zlog("Cluster aggregation complete.")
        except Exception as e:
            if debug:
                raise
            warnings.warn(f"[ZMAP] Cluster aggregation failed: {e}", stacklevel=2)

    # ------------------------------------------------------------------
    # 8. Copy colormap dict from reference → query (order-independent)
    # ------------------------------------------------------------------
    cmap_src_key = _COLORMAP_UNS_KEY.get(ref_label_col)
    if cmap_src_key and cmap_src_key in adata_ref.uns:
        cmap_dict = dict(adata_ref.uns[cmap_src_key])
        # Store under both base name and _predicted name so downstream
        # palette lookups find it regardless of which key they strip to.
        adata_query.uns[f"{ref_label_col}_color_map"] = cmap_dict
        adata_query.uns[f"{ref_label_col}_predicted_color_map"] = cmap_dict
    else:
        # Fallback: copy positional _colors arrays (legacy path)
        for key in ("ZMAP_CellType_colors", "ZMAP_Tissue_colors", "ZMAP_GermLayer_colors"):
            if key in adata_ref.uns:
                adata_query.uns[key] = adata_ref.uns[key].copy()
        warnings.warn(
            f"[ZMAP] No _color_map dict found for ref_label_col='{ref_label_col}' "
            f"(looked for uns['{cmap_src_key}']). Falling back to positional _colors "
            "arrays — palette may be incorrect if query categories differ from reference.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 9. UMAP overlay figure (v >= 2)
    # ------------------------------------------------------------------
    if _plot_basis is None:
        if v >= 1:
            _zlog(
                "Skipping UMAP overlay: no embedding found "
                "(do_ingest=False and no pre-existing X_umap)."
            )
    elif v >= 2:
        try:
            if v >= 1:
                _zlog("Plotting UMAP overlay with predicted labels...")
            plot_embedding_with_ondata_labels(
                adata_ref if _show_ref else None,
                adata_query,
                color_key=f"{space}_predicted",
                time_key=time_col_actual,
                basis=_plot_basis,
                ref_basis="X_umap",
                show_ref=_show_ref,
                show=True,
                save=save_outputs,
                output_dir=space_dir,
            )
            if v >= 1:
                _zlog("UMAP overlay figure saved.")
        except Exception as e:
            if debug:
                raise
            warnings.warn(f"[ZMAP] Failed to generate UMAP overlay figure: {e}", stacklevel=2)
    elif save_outputs:
        # Still save the figure even if not showing it
        try:
            plot_embedding_with_ondata_labels(
                adata_ref if _show_ref else None,
                adata_query,
                color_key=f"{space}_predicted",
                time_key=time_col_actual,
                basis=_plot_basis,
                ref_basis="X_umap",
                show_ref=_show_ref,
                show=False,
                save=True,
                output_dir=space_dir,
            )
            if v >= 1:
                _zlog("UMAP overlay figure saved (not displayed).")
        except Exception as e:
            if debug:
                raise
            warnings.warn(f"[ZMAP] Failed to generate UMAP overlay figure: {e}", stacklevel=2)

    # ------------------------------------------------------------------
    # 10. Label overlap heatmap (v >= 3)
    # ------------------------------------------------------------------
    if query_label_col is not None:
        show_heatmap = v >= 3
        if show_heatmap or save_outputs:
            try:
                if v >= 1:
                    _zlog(
                        f"Computing label overlap: "
                        f"'{query_label_col}' (rows) vs '{ref_label_col}' (columns)..."
                    )
                mapping_df = map_query_labels(
                    adata_query,
                    obs_A=ref_label_col+'_predicted',
                    obs_B=query_label_col,
                    normalize="row",
                    show_plot=show_heatmap,
                    return_df=True,
                    save_plots=save_outputs,
                    save_mapping=False,
                    file_prefix=space,
                    output_dir=space_dir,
                )
                adata_query.uns["zmap_labels"][space]["Label Mapping"] = mapping_df
                if v >= 1:
                    _zlog("Label overlap mapping complete.")
            except Exception as e:
                if debug:
                    raise
                warnings.warn(f"[ZMAP] Failed to compute label mapping: {e}", stacklevel=2)

    # ------------------------------------------------------------------
    # 11. Marker validation (DE overlap with ZMAP reference ledger)
    # ------------------------------------------------------------------
    if marker_validation:
        try:
            if v >= 1:
                _zlog("Validating predicted labels against ZMAP marker ledger...")
            df_markers = validate_markers(
                adata_query,
                groupby=f"{space}_predicted",
                ref_label_col=ref_label_col,
                save_csv=save_outputs,
                output_dir=space_dir,
            )
            adata_query.uns["zmap_labels"][space]["Marker Validation"] = df_markers

            if v >= 2:
                mean_overlap = df_markers["pct_overlap"].mean()
                n_groups = len(df_markers)
                n_with_overlap = int((df_markers["n_overlap"] > 0).sum())
                _zlog(
                    f"Marker validation: {n_with_overlap}/{n_groups} groups have reference overlap, "
                    f"mean {mean_overlap:.1f}% (top-20 DE vs top-100 ref)"
                )
            if v >= 3:
                _display_df(df_markers, max_rows=20)
            if v >= 1:
                _zlog("Marker validation complete.")
        except Exception as e:
            if debug:
                raise
            warnings.warn(f"[ZMAP] Marker validation failed: {e}", stacklevel=2)

    # ------------------------------------------------------------------
    # 12. Final compact summary (v >= 2)
    # ------------------------------------------------------------------
    if v >= 2:
        labels_base = f"{space}_predicted"
        n_total = int(adata_query.n_obs)
        n_assigned = int((~adata_query.obs[labels_base].isna()).sum()) if labels_base in adata_query.obs else 0
        pct = round(100.0 * n_assigned / n_total, 1) if n_total else 0.0

        # Top labels
        top_labels_str = ""
        if labels_base in adata_query.obs:
            vc = adata_query.obs[labels_base].value_counts(dropna=True)
            top3 = vc.head(3)
            parts = [f"{lbl} ({cnt:,})" for lbl, cnt in top3.items()]
            top_labels_str = ", ".join(parts)

        # Time range
        time_str = ""
        if time_col_actual in adata_query.obs:
            tvals = pd.to_numeric(adata_query.obs[time_col_actual], errors="coerce").dropna()
            if len(tvals) > 0:
                time_str = f"{tvals.min():.1f}–{tvals.max():.1f} hpf"

        _zlog(f"\n═══════════════════════════════════════════════════════")
        _zlog(f" ✓ Annotation complete — '{space}'")
        _zlog(f"   Cells: {n_assigned:,} / {n_total:,} assigned ({pct}%)")
        if top_labels_str:
            _zlog(f"   Top labels: {top_labels_str}")
        if time_str:
            _zlog(f"   Time range: {time_str}")
        # Marker validation summary
        marker_store = adata_query.uns.get("zmap_labels", {}).get(space, {}).get("Marker Validation")
        if marker_store is not None and len(marker_store) > 0:
            mean_ov = marker_store["pct_overlap"].mean()
            _zlog(f"   Marker overlap: {mean_ov:.1f}% mean (top-20 DE vs top-100 ref)")
        _zlog(f"   Outputs: {space_dir}/")
        _zlog(f"   Access:  adata.uns['zmap_labels']['{space}']")
        if query_label_col is not None:
            _zlog(f"   Clusters: adata.uns['zmap_labels']['{space}']['Cluster Summary']")
        _zlog(f"═══════════════════════════════════════════════════════")
    elif v >= 1:
        _zlog(f"✓ Annotation complete. Results in '{space_dir}/'.")

    return adata_query


# ================================================================
#  7b. Marker validation
# ================================================================

def validate_markers(
    adata_query: ad.AnnData,
    groupby: str,
    *,
    ref_label_col: str = "ZMAP_CellType",
    n_query_markers: int = 20,
    n_ref_markers: int = 100,
    max_cells_per_group: int = 2000,
    method: str = "wilcoxon",
    corr_method: str = "benjamini-hochberg",
    min_log2fc: float = 1.0,
    max_qval: float = 0.05,
    save_csv: bool = False,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Validate predicted annotations by comparing DE markers to the ZMAP reference ledger.

    For each group in ``groupby``, discovers the top DE markers via
    ``sc.tl.rank_genes_groups``, then measures overlap with the ZMAP
    consensus marker genes for the corresponding cell type. High overlap
    indicates that the transferred labels are producing biologically
    coherent groups.

    Can be called with any ``groupby`` column — use with predicted labels
    to validate the pipeline, or with ground-truth labels to compare
    annotation quality.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Annotated query dataset with log-normalized values in ``.X``.
    groupby : str
        Column in ``adata_query.obs`` to group cells by for DE testing.
        Typically ``f"{label_space}_predicted"`` or a ground-truth column.
    ref_label_col : str, default ``"ZMAP_CellType"``
        Reference label column, used to determine which ZMAP marker ledger
        to load (e.g. ``"ZMAP_CellType"`` → ``"CellType"`` level).
    n_query_markers : int, default ``20``
        Number of top DE genes to consider per group.
    n_ref_markers : int, default ``100``
        Number of top reference markers to load per group from the ZMAP
        consensus ledger.
    max_cells_per_group : int, default ``2000``
        Maximum cells per group for DE testing. Groups exceeding this are
        randomly subsampled for speed.
    method : str, default ``"wilcoxon"``
        Statistical test for ``sc.tl.rank_genes_groups``.
    corr_method : str, default ``"benjamini-hochberg"``
        Multiple testing correction method.
    min_log2fc : float, default ``1.0``
        Minimum log2 fold-change to consider a gene a valid DE marker.
    max_qval : float, default ``0.05``
        Maximum adjusted p-value to consider a gene significant.
    save_csv : bool, default ``False``
        Save the results DataFrame to ``{output_dir}/``.
    output_dir : str or None, default ``None``
        Output directory for saved CSV. Required when ``save_csv=True``.

    Returns
    -------
    pd.DataFrame
        One row per group with columns:

        - ``group``              — group label.
        - ``n_cells``            — number of cells in group.
        - ``n_de_genes``         — DE genes passing filters (up to ``n_query_markers``).
        - ``n_ref_markers``      — reference markers available for this group.
        - ``n_overlap``          — genes in both query DE and reference sets.
        - ``pct_overlap``        — ``n_overlap / n_de_genes * 100``.
        - ``overlapping_genes``  — comma-separated list of overlapping gene names.

    Examples
    --------
    Validate predicted labels (called automatically by the pipeline):

    >>> df = zmap.predict.validate_markers(adata, groupby="ZMAP_CellType_predicted")

    Compare predicted vs manual labels for a paper figure:

    >>> df_pred  = zmap.predict.validate_markers(adata, groupby="ZMAP_CellType_predicted")
    >>> df_truth = zmap.predict.validate_markers(adata, groupby="manual_annotation")
    """
    from zmap.reference.markers import load_consensus_markers

    # ---- Resolve marker level: infer from groupby, fall back to ref_label_col ----
    # Try to detect level from groupby column name (e.g. "ZMAP_Tissue_predicted" → "Tissue")
    _inferred_ref = None
    for _key in _MARKER_LEVEL_KEY:
        if groupby.startswith(_key):
            _inferred_ref = _key
            break
    _effective_ref = _inferred_ref or ref_label_col
    marker_level = _MARKER_LEVEL_KEY.get(_effective_ref)
    if marker_level is None:
        # Last resort: strip "ZMAP_" and trailing suffixes
        marker_level = _effective_ref.replace("ZMAP_", "").replace("_predicted", "").replace("_truth", "")
    _zlog(f"Loading ZMAP reference markers (level={marker_level!r}, top {n_ref_markers})...")
    ref_markers = load_consensus_markers(
        level=marker_level,
        n_per_group=n_ref_markers,
        format="sets",
    )
    _zlog(f"Loaded markers for {len(ref_markers)} groups.")

    # ---- Validate groupby column ----
    if groupby not in adata_query.obs.columns:
        raise KeyError(f"groupby column '{groupby}' not found in adata_query.obs.")

    # ---- Subsample large groups for speed ----
    obs_col = adata_query.obs[groupby].dropna()
    groups = obs_col.unique()

    if max_cells_per_group is not None:
        group_counts = obs_col.value_counts()
        oversized = group_counts[group_counts > max_cells_per_group].index
        if len(oversized) > 0:
            keep_idx = []
            for g in groups:
                g_idx = obs_col[obs_col == g].index
                if g in oversized:
                    rng = np.random.default_rng(42)
                    g_idx = rng.choice(g_idx, size=max_cells_per_group, replace=False)
                keep_idx.extend(g_idx)
            adata_sub = adata_query[keep_idx].copy()
            _zlog(f"Subsampled {len(oversized)} large groups to {max_cells_per_group} cells each.")
        else:
            adata_sub = adata_query[obs_col.index].copy()
    else:
        adata_sub = adata_query[obs_col.index].copy()

    # ---- Run DE ----
    _zlog(f"Running {method} DE test on {len(groups)} groups...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        sc.tl.rank_genes_groups(
            adata_sub,
            groupby=groupby,
            method=method,
            corr_method=corr_method,
            use_raw=False,
        )

    # ---- Extract and filter DE results per group ----
    records = []
    for g in sorted(groups, key=str):
        try:
            df_de = sc.get.rank_genes_groups_df(adata_sub, group=str(g))
        except Exception:
            continue

        # Filter by significance and fold-change
        if "logfoldchanges" in df_de.columns:
            df_de = df_de[df_de["logfoldchanges"] >= min_log2fc]
        if "pvals_adj" in df_de.columns:
            df_de = df_de[df_de["pvals_adj"] <= max_qval]

        top_genes = set(df_de.head(n_query_markers)["names"].astype(str).tolist())
        n_cells = int((obs_col == g).sum())

        # ---- Overlap with reference ----
        ref_set = ref_markers.get(str(g), set())
        overlap = top_genes & ref_set

        pct = round(100.0 * len(overlap) / len(top_genes), 1) if len(top_genes) > 0 else 0.0

        records.append({
            "group": g,
            "n_cells": n_cells,
            "n_de_genes": len(top_genes),
            "n_ref_markers": len(ref_set),
            "n_overlap": len(overlap),
            "pct_overlap": pct,
            "overlapping_genes": ", ".join(sorted(overlap)) if overlap else "",
        })

    df_result = pd.DataFrame(records)

    if save_csv and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        safe_groupby = groupby.replace("/", "_").replace(" ", "_")
        csv_path = os.path.join(output_dir, f"{safe_groupby}_marker_validation.csv")
        df_result.to_csv(csv_path, index=False)
        _zlog(f"Saved marker validation → {csv_path}")

    return df_result


def plot_marker_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    label_a: str = "Predicted",
    label_b: str = "Ground truth",
    title: str = "Marker overlap with ZMAP reference",
    sort_by: str = "a",
    min_cells: int = 0,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
) -> None:
    """
    Paired horizontal bar chart comparing marker validation results.

    Merges two ``validate_markers`` DataFrames on the ``group`` column and
    plots ``pct_overlap`` side by side for each cell type.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        Output of ``validate_markers`` for two different ``groupby`` columns
        (e.g. predicted labels vs ground-truth annotations).
    label_a, label_b : str
        Legend labels for the two sets of bars.
    title : str
        Plot title.
    sort_by : str, default ``"a"``
        Sort groups by: ``"a"`` (df_a overlap descending), ``"b"``,
        ``"diff"`` (a − b descending), or ``"name"`` (alphabetical).
    min_cells : int, default ``0``
        Exclude groups with fewer than this many cells in either set.
    figsize : tuple or None
        Figure size. Auto-scaled to number of groups when ``None``.
    save : str or None
        Path to save the figure. ``None`` to skip saving.

    Examples
    --------
    >>> df_pred  = zmap.predict.validate_markers(adata, groupby="ZMAP_CellType_predicted")
    >>> df_truth = zmap.predict.validate_markers(adata, groupby="manual_annotation")
    >>> zmap.predict.plot_marker_comparison(df_pred, df_truth)
    """
    # Merge on group
    merged = pd.merge(
        df_a[["group", "n_cells", "pct_overlap"]].rename(
            columns={"pct_overlap": "pct_a", "n_cells": "n_cells_a"}
        ),
        df_b[["group", "n_cells", "pct_overlap"]].rename(
            columns={"pct_overlap": "pct_b", "n_cells": "n_cells_b"}
        ),
        on="group",
        how="outer",
    ).fillna(0)

    # Filter by min_cells
    if min_cells > 0:
        merged = merged[
            (merged["n_cells_a"] >= min_cells) & (merged["n_cells_b"] >= min_cells)
        ]

    # Sort
    if sort_by == "a":
        merged = merged.sort_values("pct_a", ascending=True)
    elif sort_by == "b":
        merged = merged.sort_values("pct_b", ascending=True)
    elif sort_by == "diff":
        merged = merged.assign(_diff=merged["pct_a"] - merged["pct_b"])
        merged = merged.sort_values("_diff", ascending=True).drop(columns=["_diff"])
    else:
        merged = merged.sort_values("group", ascending=False)

    n = len(merged)
    if n == 0:
        _zlog("No overlapping groups to plot.")
        return

    if figsize is None:
        figsize = (6, max(3, n * 0.35))

    y = np.arange(n)
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y + bar_h / 2, merged["pct_a"].values, bar_h, label=label_a, color="steelblue", alpha=0.85)
    ax.barh(y - bar_h / 2, merged["pct_b"].values, bar_h, label=label_b, color="coral", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(merged["group"].values, fontsize=8)
    ax.set_xlabel("% overlap (top-20 DE ∩ top-100 ref)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, max(merged[["pct_a", "pct_b"]].max().max() * 1.15, 10))

    # Mean lines
    mean_a = merged["pct_a"].mean()
    mean_b = merged["pct_b"].mean()
    ax.axvline(mean_a, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(mean_b, color="coral", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
        _zlog(f"Saved comparison figure → {save}")

    plt.show()


# ================================================================
#  8. On-demand plot accessors
# ================================================================

def _resolve_config(
    adata_query: ad.AnnData,
    label_space: str | None,
) -> tuple[str, dict]:
    """Resolve label_space from _last_space if needed, return (space, config)."""
    store = adata_query.uns.get("zmap_labels", {})
    if label_space is None:
        label_space = store.get("_last_space")
        if label_space is None:
            raise KeyError(
                "No label_space provided and no '_last_space' found in "
                "adata.uns['zmap_labels']. Run annotate_with_zmap first."
            )
    space_store = store.get(label_space, {})
    config = space_store.get("_run_config", {})
    return label_space, config


def plot_qc(
    adata_query: ad.AnnData,
    label_space: str | None = None,
    *,
    save: bool = False,
) -> None:
    """
    Re-plot the combined probability/distance QC histograms from a completed run.

    All parameters are resolved automatically from the stored run config.
    Just call ``zmap.predict.plot_qc(adata)``.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Annotated query dataset (must have been processed by ``annotate_with_zmap``).
    label_space : str or None, default ``None``
        Label namespace. When ``None``, uses the most recent run.
    save : bool, default ``False``
        Save the figure to the run's output directory.
    """
    label_space, config = _resolve_config(adata_query, label_space)
    labels_base = f"{label_space}_predicted"
    col_prob = f"{labels_base}_prob"
    col_dist = f"{labels_base}_dist"

    for col in (col_prob, col_dist):
        if col not in adata_query.obs.columns:
            raise KeyError(
                f"Column '{col}' not found in adata_query.obs. "
                f"Run annotate_with_zmap with label_space='{label_space}' first."
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.hist(adata_query.obs[col_prob].dropna(), bins=100, color='steelblue', alpha=0.7)
    ax1.set_title("Predicted Probability")
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("Cell Count")

    ax2.hist(adata_query.obs[col_dist].dropna(), bins=100, color='steelblue', alpha=0.7)
    ax2.set_title("Median Neighbor Distance")
    ax2.set_xlabel("Neighbor Distance")
    ax2.set_ylabel("Cell Count")

    fig.tight_layout()

    if save:
        space_dir = config.get("output_dir", os.path.join("zmap_predict", label_space))
        os.makedirs(space_dir, exist_ok=True)
        qc_path = os.path.join(space_dir, f"{labels_base}_qc_summary.png")
        fig.savefig(qc_path, dpi=300, bbox_inches="tight")
        _zlog(f"Saved QC plot: {qc_path}")

    plt.show()


def plot_embedding(
    adata_query: ad.AnnData,
    label_space: str | None = None,
    *,
    save: bool = False,
    show_labels: bool = True,
    ref_size: float = 2,
    ref_alpha: float = 0.3,
    test_size: float = 2,
    test_alpha: float = 1.0,
    return_ax: bool = False,
    **kwargs,
) -> tuple | None:
    """
    Re-plot the UMAP overlay with on-data labels and time strip.

    Reproduces the UMAP figure from the pipeline using the stored reference
    UMAP coordinates and run config. No ``adata_ref`` needed.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Annotated query dataset (must have been processed by ``annotate_with_zmap``).
    label_space : str or None, default ``None``
        Label namespace. When ``None``, uses the most recent run.
    save : bool, default ``False``
        Save the figure to the run's output directory.
    show_labels : bool, default ``True``
        If ``True``, draw on-data text labels at category centroids.
        If ``False``, suppress all text labels and arrows.
    ref_size : float, default ``2``
        Scatter point size for reference background cells.
    ref_alpha : float, default ``0.3``
        Opacity of reference background points.
    test_size : float, default ``2``
        Scatter point size for query (projected) cells.
    test_alpha : float, default ``1.0``
        Opacity of query overlay points.
    return_ax : bool, default ``False``
        Return ``(fig, ax_umap, ax_strip)`` instead of ``None``.
    **kwargs
        Extra keyword arguments forwarded to ``plot_embedding_with_ondata_labels``.

    Returns
    -------
    tuple or None
        ``(fig, ax_umap, ax_strip)`` when ``return_ax=True``, otherwise ``None``.
    """
    label_space, config = _resolve_config(adata_query, label_space)
    space_store = adata_query.uns.get("zmap_labels", {}).get(label_space, {})

    # Resolve basis and show_ref from stored config
    plot_basis = config.get("basis", "X_umap_zmap")
    show_ref = config.get("show_ref", True)

    if plot_basis is None:
        raise KeyError(
            f"No embedding basis stored for label_space='{label_space}'. "
            "The original run had do_ingest=False and no pre-existing X_umap."
        )

    # Build reference shell only when needed
    adata_ref_mini = None
    if show_ref:
        ref_umap = space_store.get("_ref_umap")
        if ref_umap is None:
            raise KeyError(
                f"No stored reference UMAP found at "
                f"adata.uns['zmap_labels']['{label_space}']['_ref_umap']. "
                "Run annotate_with_zmap first."
            )
        adata_ref_mini = ad.AnnData(
            obs=pd.DataFrame(index=[f"ref_{i}" for i in range(ref_umap.shape[0])]),
        )
        adata_ref_mini.obsm["X_umap"] = ref_umap

    color_key = f"{label_space}_predicted"
    time_key = config.get("time_col", "ZMAP_time_id_predicted")
    space_dir = config.get("output_dir", os.path.join("zmap_predict", label_space))

    return plot_embedding_with_ondata_labels(
        adata_ref_mini,
        adata_query,
        color_key=color_key,
        time_key=time_key,
        basis=plot_basis,
        ref_basis="X_umap",
        show_ref=show_ref,
        show=True,
        save=save,
        show_labels=show_labels,
        ref_size=ref_size,
        ref_alpha=ref_alpha,
        test_size=test_size,
        test_alpha=test_alpha,
        return_ax=return_ax,
        output_dir=space_dir,
        **kwargs,
    )


def plot_time(
    adata_query: ad.AnnData,
    label_space: str | None = None,
    *,
    save: bool = False,
    **kwargs,
) -> None:
    """
    Plot the predicted time distribution as a standalone colorbar histogram.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Annotated query dataset (must have been processed by ``annotate_with_zmap``).
    label_space : str or None, default ``None``
        Label namespace. When ``None``, uses the most recent run.
    save : bool, default ``False``
        Save the figure to the run's output directory.
    **kwargs
        Extra keyword arguments forwarded to ``plot_colorbar_histogram``.
    """
    label_space, config = _resolve_config(adata_query, label_space)
    time_col = config.get("time_col", "ZMAP_time_id_predicted")

    if time_col not in adata_query.obs.columns:
        raise KeyError(
            f"Column '{time_col}' not found in adata_query.obs. "
            "Run annotate_with_zmap first."
        )

    fig, ax = plt.subplots(figsize=(8, 0.6))
    plot_colorbar_histogram(
        adata_query.obs[time_col],
        ax=ax,
        **kwargs,
    )

    if save:
        space_dir = config.get("output_dir", os.path.join("zmap_predict", label_space))
        os.makedirs(space_dir, exist_ok=True)
        path = os.path.join(space_dir, f"{label_space}_time_distribution.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        _zlog(f"Saved time plot: {path}")

    plt.show()


def plot_overlap_matrix(
    adata_query: ad.AnnData,
    label_space: str | None = None,
    *,
    save: bool = False,
    **kwargs,
) -> pd.DataFrame | None:
    """
    Re-plot the query_label_col × ZMAP label overlap matrix.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Annotated query dataset (must have been processed by ``annotate_with_zmap``).
    label_space : str or None, default ``None``
        Label namespace. When ``None``, uses the most recent run.
    save : bool, default ``False``
        Save the figure to the run's output directory.
    **kwargs
        Extra keyword arguments forwarded to ``map_query_labels``.

    Returns
    -------
    pd.DataFrame
        Per-group best-match mapping table.
    """
    label_space, config = _resolve_config(adata_query, label_space)

    query_label_col = config.get("query_label_col")
    if query_label_col is None:
        raise ValueError(
            "No query_label_col was set during the pipeline run. "
            "Re-run annotate_with_zmap with query_label_col='...' to enable "
            "the overlap matrix."
        )

    obs_A = f"{label_space}_predicted"
    space_dir = config.get("output_dir", os.path.join("zmap_predict", label_space))

    return map_query_labels(
        adata_query,
        obs_A=obs_A,
        obs_B=query_label_col,
        normalize="row",
        show_plot=True,
        return_df=True,
        save_plots=save,
        save_mapping=False,
        file_prefix=label_space,
        output_dir=space_dir,
        **kwargs,
    )


def show_summary(
    adata_query: ad.AnnData,
    label_space: str | None = None,
) -> None:
    """
    Re-display the run summary and annotation tables from a completed run.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Annotated query dataset (must have been processed by ``annotate_with_zmap``).
    label_space : str or None, default ``None``
        Label namespace. When ``None``, uses the most recent run.
    """
    label_space, config = _resolve_config(adata_query, label_space)
    store = adata_query.uns.get("zmap_labels", {}).get(label_space, {})
    if not store:
        raise KeyError(
            f"No results found at adata.uns['zmap_labels']['{label_space}']. "
            "Run annotate_with_zmap first."
        )

    if "Run Summary Simple" in store:
        _zlog(f"── Run Summary ({label_space}) ──────────────────────────")
        _display_df(store["Run Summary Simple"])

    if "Cell Annotations" in store:
        df = store["Cell Annotations"]
        _zlog(f"\n── Cell Annotations ({len(df):,} cells) ──────────────────")
        _display_df(df, max_rows=10)

    if "Cluster Summary" in store:
        df = store["Cluster Summary"]
        _zlog(f"\n── Cluster Summary ({len(df)} clusters) ─────────────────")
        _display_df(df, max_rows=20)

