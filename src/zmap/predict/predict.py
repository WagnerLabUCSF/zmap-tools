# -------------------------------------------------------------------
#  ZMAP  —  Prediction, kNN transfer, preprocessing, diagnostics
# -------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Mapping

import warnings, os

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


# ================================================================
#  0. Query Preprocessing (TPM → log1p stored in adata.X)
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

            if r_rows.size >= k:
                local_idx, local_dist, local_meta = _run_knn(
                    X_ref[r_rows],
                    X_query[q_rows],
                    k_use=k,
                    tag=f"local|{str(tissue)}",
                )
                if first_meta is None:
                    first_meta = local_meta
                mapped = np.full_like(local_idx, -1)
                ok = local_idx >= 0
                if np.any(ok):
                    mapped[ok] = r_rows[local_idx[ok]]
                idx_out[q_rows] = mapped
                dist_out[q_rows] = local_dist
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


def predict_label_tissue_kNN(
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
    # --- backend controls (aligned with knn_backend.py) ---
    knn_backend: str = "auto",           # "auto" | "faiss" | "sklearn"
    knn_device: str = "auto",            # "auto" | "cpu" | "cuda" | "cuda:N"
    knn_nprobe: int | None = None,
    knn_l2norm: bool = False,
    # --- compatibility params from tissue-aware API ---
    class_prior_alpha: float = 0.0,
    pseudo_tissue_k: int | None = None,
    pseudo_tissue_threshold: float = 0.0,
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
    p_thresh: float | None = 0.8,
    d_thresh: float | None = 0.1,
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

    if pseudo_tissue_k is not None or float(pseudo_tissue_threshold) > 0:
        print(
            "[ZMAP] predict_label_tissue_kNN: pseudo_tissue_k / pseudo_tissue_threshold "
            "are accepted for API compatibility but not used in this function."
        )
    if float(class_prior_alpha) != 0.0:
        print(
            "[ZMAP] predict_label_tissue_kNN: class_prior_alpha is accepted for "
            "API compatibility but not used in predict_labels_kNN voting."
        )
    if float(margin_threshold) > 0:
        print(
            "[ZMAP] predict_label_tissue_kNN: margin_threshold is accepted for API "
            "compatibility but not applied in predict_labels_kNN."
        )
    if bool(run_time_prediction):
        print(
            "[ZMAP] predict_label_tissue_kNN: run_time_prediction/time_* parameters "
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
        int(n_ref_keep),
        int(np.sum(np.flatnonzero(ref_keep_mask.to_numpy()) % 1048573)),
    ]

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

        ref_tissue = None
        query_tissue = None
        if mode != "none":
            if ref_tissue_col not in adata_ref.obs:
                raise KeyError(
                    f"Missing tissue column in adata_ref.obs: {ref_tissue_col} "
                    "(required for tissue_mode='hard'/'soft')."
                )
            if query_tissue_col not in adata_query.obs:
                raise KeyError(
                    f"Missing tissue column in adata_query.obs: {query_tissue_col} "
                    "(required for tissue_mode='hard'/'soft')."
                )
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
            "knn_l2norm": bool(knn_l2norm),
            "knn_backend_requested": knn_meta.get("backend_requested", knn_backend),
            "knn_device_requested": knn_meta.get("device_requested", knn_device),
            "knn_nprobe_requested": (None if knn_nprobe is None else int(knn_nprobe)),
            "knn_backend_used": knn_meta.get("backend_used", "sklearn"),
            "knn_device_used": knn_meta.get("device_used", "cpu"),
        }
    else:
        print("Reusing cached tissue-aware neighbor graph from adata_query.uns['zmap_neighbors'].")

    p_thresh_use = p_thresh
    if confidence_threshold is not None:
        p_thresh_use = float(confidence_threshold)

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
        p_thresh=p_thresh_use,
        d_thresh=d_thresh,
        min_cells_per_label=min_cells_per_label,
        apply_filters=apply_filters,
        output_dir=output_dir,
        expected_cache_mode=mode,
    )

    space = label_space or ref_label_col
    adata_query.uns.setdefault("zmap_labels", {}).setdefault(space, {})
    adata_query.uns["zmap_labels"][space]["Tissue-aware kNN"] = {
        "tissue_mode": mode,
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
        "class_prior_alpha": float(class_prior_alpha),
        "pseudo_tissue_k": (None if pseudo_tissue_k is None else int(pseudo_tissue_k)),
        "pseudo_tissue_threshold": float(pseudo_tissue_threshold),
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
#  1. Predict labels via filtered kNN (your full function)
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

    # QC thresholds
    p_thresh: float | None = 0.8,
    d_thresh: float | None = 0.1,
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
    p_thresh : float or None, default ``0.8``
        Minimum vote probability required to assign a label. Cells below this
        threshold are marked as unassigned.
    d_thresh : float or None, default ``0.1``
        Maximum allowable mean distance to neighbors. Cells exceeding this
        threshold are marked as low-confidence.
    min_cells_per_label : int, default ``15``
        Minimum number of reference cells a label must have to be included in
        voting. Labels with fewer cells are treated as ``omit_labels``.
    apply_filters : bool, default ``True``
        Apply ``p_thresh`` and ``d_thresh`` filters to produce the final
        predicted label column. Set to ``False`` to retain raw predictions.

    Returns
    -------
    None
        Results are written directly into ``adata_query``:

        - ``adata_query.obs[f"{label_space}_predicted"]`` — predicted labels.
        - ``adata_query.obs[f"{label_space}_prob"]``      — top-label vote probability.
        - ``adata_query.obs["ZMAP_time_id"]``             — predicted developmental time.
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
    if time_distance not in (None, "gaussian", "inverse"):
        raise ValueError("time_distance must be one of {None, 'gaussian', 'inverse'}.")
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
    _mask_digest = [int(n_ref_keep), int(np.sum(np.flatnonzero(ref_keep_mask.to_numpy()) % 1048573))]

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
            cache.get('mask_digest') == _mask_digest and
            cache.get('tissue_mode', 'none') == expected_cache_mode and
            cache.get('knn_backend_requested', 'auto') == knn_backend and
            cache.get('knn_device_requested', 'auto') == knn_device and
            cache.get('knn_nprobe_requested', None) == (None if knn_nprobe is None else int(knn_nprobe))
        )
        if same_config:
            print("Reusing cached neighbor graph from adata_query.uns['zmap_neighbors'] (filtered).")
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
        print("Computing new kNN graph on filtered reference...")
        neighbor_indices, distances, knn_meta = knn_search(
            X_ref,
            adata_query.obsm[query_basis],
            n_neighbors=n_neighbors,
            metric=metric,
            backend=knn_backend,
            device=knn_device,
            nprobe=knn_nprobe,
        )
        print(
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
        print("Cached neighbor graph saved in adata_query.uns['zmap_neighbors'].")

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

    for i, classes in enumerate(neighbor_classes):
        mask = ~pd.isna(classes)
        if not np.any(mask):
            continue
        vals = np.asarray(classes[mask], dtype=str)
        idxs = np.searchsorted(sorted_classes, vals)
        if class_balance == "global_inverse":
            weights = w_class[idxs]
            scores = np.bincount(idxs, weights=weights, minlength=C)
        else:
            scores = np.bincount(idxs, minlength=C)
        s = scores.sum()
        if s > 0:
            probabilities_sorted[i, :] = scores / s

    predicted_labels = sorted_classes[np.argmax(probabilities_sorted, axis=1)]

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

    print("Predictions complete.")

    # =======================
    #     QC FILTERING
    # =======================
    accept = pd.Series(True, index=adata_query.obs.index)
    if apply_filters:
        use_prob = (p_thresh is not None)
        use_dist = (d_thresh is not None)

        p_ok = (adata_query.obs[col_prob] >= p_thresh).fillna(False) if use_prob else None
        d_ok = (adata_query.obs[col_dist] <= d_thresh).fillna(False) if use_dist else None

        if not use_prob and not use_dist:
            accept = pd.Series(True, index=adata_query.obs.index)
            reason = np.array(["auto"] * len(accept), dtype=object)
            reason_categories = ["auto"]
            print("QC skipped: p_thresh=None and d_thresh=None → accepting all cells.")
        else:
            accept = pd.Series(False, index=adata_query.obs.index)
            if use_prob:
                accept |= p_ok
            if use_dist:
                accept |= d_ok

            if use_prob and use_dist:
                reason = np.where(p_ok & d_ok, "both",
                          np.where(p_ok, "proba",
                          np.where(d_ok, "distance", "none")))
                reason_categories = ["proba", "distance", "both", "none"]
            elif use_prob:
                reason = np.where(p_ok, "proba", "none")
                reason_categories = ["proba", "none"]
            else:
                reason = np.where(d_ok, "distance", "none")
                reason_categories = ["distance", "none"]

            print(f"QC applied with active rules: "
                  f"{'prob' if use_prob else ''}{' & ' if use_prob and use_dist else ''}{'distance' if use_dist else ''}.")

        adata_query.obs[col_reason] = pd.Categorical(reason, categories=reason_categories)
        adata_query.obs[col_reject] = ~accept

        # Mask rejected predictions
        adata_query.obs.loc[~accept, col_main] = pd.NA

        n_total = len(accept)
        n_accept = int(accept.sum())
        print(f"{n_accept} accepted / {n_total} total ({n_total - n_accept} rejected).")

        # ---------- QC PLOTTING & OPTIONAL SAVE ----------
        if plot_mapping_qc:
            import os

            os.makedirs(output_dir, exist_ok=True)

            # ----- Probability histogram -----
            fig1 = plt.figure()
            plt.hist(adata_query.obs[col_prob].dropna(), bins=100,
                    color='steelblue', alpha=0.7)
            if use_prob:
                plt.axvline(p_thresh, color='red', linestyle='--',
                            label=f'p_thresh={p_thresh}')
            plt.title(f"Predicted Probability\n{n_accept} pass / {n_total} total")
            plt.xlabel('Predicted Probability')
            plt.ylabel('Cell Count')

            if plt.gca().get_legend_handles_labels()[1]:
                plt.legend()

            plt.tight_layout()

            # save figure
            if save_mapping_qc:
                prob_path = os.path.join(output_dir, f"{labels_base}_qc_probability.png")
                fig1.savefig(prob_path, dpi=300)
                print(f"[ZMAP] Saved QC plot: {prob_path}")

            plt.show()

            # ----- Distance histogram -----
            fig2 = plt.figure()
            plt.hist(adata_query.obs[col_dist].dropna(), bins=100,
                    color='steelblue', alpha=0.7)
            if use_dist:
                plt.axvline(d_thresh, color='red', linestyle='--',
                            label=f'd_thresh={d_thresh}')
            plt.title(f"{metric.title()} Median Distance\n{n_accept} pass / {n_total} total")
            plt.xlabel('Neighbor Distance')
            plt.ylabel('Cell Count')

            if plt.gca().get_legend_handles_labels()[1]:
                plt.legend()

            plt.tight_layout()

            # save figure
            if save_mapping_qc:
                dist_path = os.path.join(output_dir, f"{labels_base}_qc_distance.png")
                fig2.savefig(dist_path, dpi=300)
                print(f"[ZMAP] Saved QC plot: {dist_path}")

            plt.show()


    # ---------- rare label filter ----------
    if min_cells_per_label is not None and min_cells_per_label > 0:
        print(f"Filtering labels with fewer than {min_cells_per_label} assigned cells...")
        label_counts = adata_query.obs[col_main].value_counts(dropna=True)
        rare_labels = label_counts[label_counts < min_cells_per_label].index
        if len(rare_labels) > 0:
            adata_query.obs[col_rareflag] = adata_query.obs[col_main].isin(rare_labels)
            adata_query.obs.loc[adata_query.obs[col_rareflag], col_main] = pd.NA
            adata_query.uns.setdefault('zmap_labels', {}).setdefault(space, {})
            adata_query.uns['zmap_labels'][space]['Rare Labels'] = list(rare_labels)
            print(f"Filtered {len(rare_labels)} rare labels: {list(rare_labels[:10])}{'...' if len(rare_labels) > 10 else ''}")

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

    # ---------- evaluation (unchanged) ----------
    if evaluate:
        if not query_truth_col or query_truth_col not in adata_query.obs.columns:
            print(f"Evaluation skipped: ground-truth column '{query_truth_col}' not found in adata_query.obs.")
            print(f"Finished predicting and annotating: {space}")
            return

        print("Evaluating model performance on ACCEPTED predictions only...")
        has_truth = ~adata_query.obs[query_truth_col].isna()
        not_rejected = (~adata_query.obs[col_reject].fillna(True)) if (apply_filters and col_reject in adata_query.obs) else True
        has_pred = ~adata_query.obs[col_main].isna()
        eval_mask = has_truth & not_rejected & has_pred

        n_eval = int(eval_mask.sum())
        if n_eval == 0:
            print("No accepted rows available for evaluation after filtering; metrics not computed.")
            print(f"Finished predicting and annotating: {space}")
            return

        true_labels_values      = adata_query.obs.loc[eval_mask, query_truth_col].astype(str).values
        predicted_labels_values = adata_query.obs.loc[eval_mask, col_main].astype(str).values
        probabilities_eval      = adata_query.obsm[col_probs_mat][eval_mask, :]

        true_classes = set(np.unique(true_labels_values))
        predicted_classes = set(np.unique(predicted_labels_values))
        overlapping_classes = sorted(true_classes.intersection(predicted_classes))
        if len(overlapping_classes) == 0:
            print("No overlapping classes between true and predicted after filtering; metrics not computed.")
            print(f"Finished predicting and annotating: {space}")
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

        metrics_dict = {
            "Aggregate Metrics": pd.DataFrame({
                "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro AUROC"],
                "Score": [accuracy, macro_precision, macro_recall, macro_f1, macro_auroc],
            }),
            "Class-Specific Metrics": pd.DataFrame({
                "Class": overlapping_classes,
                "Precision": per_class[0],
                "Recall": per_class[1],
                "F1-Score": per_class[2],
                "AUROC": [class_auroc[label] for label in overlapping_classes],
                "Support": per_class[3],
            }),
            "Confusion Matrix": cm_df,
            "Eval N": n_eval,
        }

        metrics_dict["Run Summary"] = adata_query.uns['zmap_labels'].get(space, {}).get("Run Summary")
        adata_query.uns.setdefault('zmap_labels', {})
        adata_query.uns['zmap_labels'][space] = metrics_dict

        print(metrics_dict["Aggregate Metrics"])

        if plot_eval_curves:
            print("Plotting ROC and PR curves...")
            for i, label in enumerate(overlapping_classes):
                plt.figure(figsize=(8, 4))
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], probabilities_eval[:, i])
                precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], probabilities_eval[:, i])

                plt.subplot(1, 2, 1); plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.2f}"); plt.plot([0, 1], [0, 1], 'k--')
                plt.title(f"ROC – {label}"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.legend()

                plt.subplot(1, 2, 2); plt.plot(recall, precision)
                plt.title(f"Precision–Recall – {label}"); plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.tight_layout(); plt.show()

    print(f"Finished predicting and annotating: {space}")


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

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{label_space}_cluster_summary.csv")
        df.to_csv(out_path, index=False)
        print(f"[ZMAP] Saved cluster summary → {out_path}")

    return df


# ================================================================
#  4. Build per-cell annotation table
# ================================================================

def build_cell_annotations_table(
    adata_query: ad.AnnData,
    label_space: str,
    *,
    cluster_col: str | None = None,
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
    save_csv : bool, default ``True``
        Write the table to
        ``./zmap/predict/{label_space}_cell_annotations.csv``.

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
        - ``ZMAP_time_id``              — predicted developmental time (hpf).
    """
    labels_base = f"{label_space}_predicted"
    col_main   = labels_base
    col_prob   = f"{labels_base}_prob"
    col_reject = f"{labels_base}_reject_flag"
    col_reason = f"{labels_base}_reason"
    time_col   = "ZMAP_time_id"

    wanted: list[str] = []
    if cluster_col and cluster_col in adata_query.obs.columns:
        wanted.append(cluster_col)
    for col in [col_main, col_prob, col_reject, col_reason, time_col]:
        if col in adata_query.obs.columns:
            wanted.append(col)

    df = adata_query.obs[wanted].copy()
    df.index.name = "cell_id"
    df = df.reset_index()

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{label_space}_cell_annotations.csv")
        df.to_csv(out_path, index=False)
        print(f"[ZMAP] Saved cell annotations → {out_path}")

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

    # ---- point/legend style ----
    ref_size: float = 2,
    test_size: float = 2,
    cmap: str = "jet",
    frameon: bool = False,
    sort_order: bool = True,
    legend_loc: str = "on data",
    legend_fontsize: float = 5,
    legend_fontweight: str = "normal",

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

    Renders two layers: (1) the full reference embedding as a faint grey
    background for spatial context, and (2) the query cells colored by a
    predicted label column. Labels are drawn directly on the embedding using
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
        ``obsm`` key used for the 2D embedding coordinates in both datasets.
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
    ref_size, test_size : float, default ``2``
        Scatter point sizes for reference background and query cells respectively.
    cmap : str, default ``"jet"``
        Colormap used for the reference background scatter.
    legend_loc : str, default ``"on data"``
        Where to place the category legend. ``"on data"`` draws labels directly
        at centroid positions; other values follow matplotlib legend conventions.
    legend_fontsize, legend_fontweight : float and str, default ``5`` and ``"normal"``
        Font size and weight for on-data legend labels.
    replace_underscores : bool, default ``True``
        Replace underscores in label strings with line breaks for cleaner
        on-data annotation.
    adjust_expand : tuple of float, default ``(1.2, 1.5)``
        ``(x_expand, y_expand)`` passed to ``adjustText`` for label placement.
    match_arrow_color_to_text : bool, default ``True``
        Color annotation arrows to match their corresponding text label.
    show : bool, default ``False``
        Call ``plt.show()`` after rendering.
    save : bool, default ``True``
        Save the figure as PNG and PDF to ``./zmap/predict/``.
    return_ax : bool, default ``False``
        Return the main ``matplotlib.axes.Axes`` object.

    Returns
    -------
    matplotlib.axes.Axes or None
        The main axes when ``return_ax=True``, otherwise ``None``.
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
    try:
        # ensure reference has its color map
        sync_zmap_colors(adata_ref, obs_key=base_obs)
        # sync query colors from reference
        sync_zmap_colors(adata_test_plot, obs_key=base_obs, ref_adata=adata_ref)
    except Exception:
        # silently fall back to whatever is already in adata_test_plot.uns
        pass

    # ---- palette construction ----
    if palette is None:
        if palette_uns_key is None:
            # e.g. color_key="ZMAP_Tissue_predicted" → use "ZMAP_Tissue_colors"
            palette_uns_key = f"{base_obs}_colors"
        if palette_uns_key not in adata_test_plot.uns:
            raise KeyError(
                f"Palette not provided and '{palette_uns_key}' not found in adata.uns. "
                f"Provide `palette` or ensure `{palette_uns_key}` exists."
            )
        cats = adata_test_plot.obs[color_key].cat.categories
        colors = adata_test_plot.uns[palette_uns_key]
        palette = dict(zip(cats, colors))

    # ---- defaults for arrowprops ----
    if arrowprops is None:
        arrowprops = dict(arrowstyle='-', color='k', lw=0.25)

    ref_kwargs = {} if ref_kwargs is None else dict(ref_kwargs)
    test_kwargs = {} if test_kwargs is None else dict(test_kwargs)
    time_strip_kwargs = {} if time_strip_kwargs is None else dict(time_strip_kwargs)

    # ---- Check if we actually have a time vector ----
    has_time = (
        show_time_strip and
        (time_key in adata_test.obs.columns) and
        pd.to_numeric(adata_test.obs[time_key], errors="coerce").notna().any()
    )

    fig = None
    ax_umap = None
    ax_strip = None

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

        # ---- reference embedding in UMAP axis ----
        ax = sc.pl.embedding(
            adata_ref,
            basis=basis,
            show=False,
            s=ref_size,
            ax=ax_umap,
            **ref_kwargs,
        )

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
            legend_loc=legend_loc,
            legend_fontweight=legend_fontweight,
            ax=ax,
            show=False,
            title=color_key + " (Predicted)",
            palette=palette,
            **test_kwargs,
        )

        # --- obtain existing on-data labels ---
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
            print(f"[ZMAP] Saved figure:\n - {png_path}\n - {pdf_path}")

        # ---- SHOW FIGURE ----
        if show:
            plt.show()

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
    if show_plot:
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
        print(f"[ZMAP] Saved overlap figure → {output_dir}/{base}.png")

    if save_mapping and mapping_df is not None:
        out_csv = os.path.join(output_dir, f"{prefix}_{obs_B}_top_label.csv")
        mapping_df.to_csv(out_csv)
        print(f"[ZMAP] Saved top-label mapping → {out_csv}")

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

    # --- cluster aggregation ---
    cluster_col: str | None = None,       # user-defined clusters, e.g. "leiden"
    query_label_col: str | None = None,   # deprecated alias for cluster_col

    # --- pipeline toggles ---
    do_preprocess: bool = True,
    do_map_embedding: bool = True,
    do_ingest: bool = True,

    # --- kwargs passthroughs to lower-level steps ---
    preprocess_kwargs: Mapping[str, Any] | None = None,
    predict_kwargs: Mapping[str, Any] | None = None,

    # --- output controls ---
    print_summary: bool = True,
    show_plots: bool = True,              # set False in headless / script contexts
    save_outputs: bool = True,            # save CSVs and PNGs to output_dir
    output_dir: str = "zmap_predict",     # directory for all saved files
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
       ``predict_label_tissue_kNN``).
    4. **Summarize** — store a simplified run summary in
       ``adata_query.uns['zmap_labels'][<space>]['Run Summary Simple']``.
    5. **Plot** — overlay query cells on the reference UMAP with on-data labels
       (``plot_embedding_with_ondata_labels``).
    6. **Map labels** *(optional)* — cross-tabulate ZMAP labels against an existing
       query labeling (e.g. Leiden clusters) via ``map_query_labels``.

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
        metrics when ``predict_kwargs`` includes ``evaluate=True``.
    cluster_col : str or None, default ``None``
        Column in ``adata_query.obs`` containing user-defined cluster IDs
        (e.g. ``"leiden"``). When provided, enables cluster-level consensus
        aggregation and the label-overlap heatmap. Recommended for most workflows.
    query_label_col : str or None, default ``None``
        Deprecated alias for ``cluster_col``. Use ``cluster_col`` instead.
    do_preprocess : bool, default ``True``
        Run TPM normalization + log1p on the query before mapping.
        Set to ``False`` if ``adata_query.X`` is already log-normalized.
    do_map_embedding : bool, default ``True``
        Run Symphony embedding mapping. Requires ``symphonypy``. Set to ``False``
        if the query already has a ``X_pca_harmony`` embedding.
    do_ingest : bool, default ``True``
        Ingest the query into the reference UMAP after Symphony mapping.
        Only applies when ``do_map_embedding=True``.
    preprocess_kwargs : dict or None, default ``None``
        Extra keyword arguments forwarded to ``preprocess_adata_query``
        (e.g. ``{"strict_counts": True}``).
    predict_kwargs : dict or None, default ``None``
        Extra keyword arguments forwarded to ``predict_labels_kNN``.
        Set ``{"use_tissue_aware_knn": True}`` to route step-3 transfer to
        ``predict_label_tissue_kNN`` instead (same step-4 summary format).
    print_summary : bool, default ``True``
        Print a brief progress log and final summary to stdout.
    show_plots : bool, default ``True``
        Call ``plt.show()`` after each figure. Set to ``False`` in headless
        or script contexts (figures are still saved when ``save_outputs=True``).
    save_outputs : bool, default ``True``
        Save cell annotations CSV, cluster summary CSV, and all figures
        to ``./zmap/predict/``.

    Returns
    -------
    anndata.AnnData
        The annotated query dataset (same object, modified in-place). Key
        additions to ``adata_query``:

        - ``.obs[f"{label_space}_predicted"]``                       — transferred cell labels.
        - ``.obs[f"{label_space}_prob"]``                            — label confidence (0–1).
        - ``.obs["ZMAP_time_id"]``                                   — predicted time (hpf).
        - ``.obsm["X_umap"]``                                        — UMAP coordinates (if ingested).
        - ``.uns['zmap_labels'][label_space]['Run Summary Simple']`` — key/value run summary.
        - ``.uns['zmap_labels'][label_space]['Cell Annotations']``   — per-cell table.
        - ``.uns['zmap_labels'][label_space]['Cluster Summary']``    — cluster consensus table
          (only when ``cluster_col`` is provided).
        - ``.uns['zmap_labels'][label_space]['Label Mapping']``      — cluster×label overlap
          (only when ``cluster_col`` is provided).

    Examples
    --------
    Minimal usage:

    >>> adata = zmap.predict.annotate_with_zmap(
    ...     adata_query,
    ...     query_raw_counts_source="counts",
    ...     cluster_col="leiden",
    ... )

    With custom reference label and predict options:

    >>> adata = zmap.predict.annotate_with_zmap(
    ...     adata_query,
    ...     query_raw_counts_source="X",
    ...     ref_label_col="ZMAP_Tissue",
    ...     cluster_col="leiden",
    ...     predict_kwargs={"n_neighbors": 50},
    ... )
    """

    # Suppress UMAP "n_jobs overridden" warnings
    warnings.filterwarnings(
        "ignore",
        message=".*overridden to 1 by setting random_state.*"
    )

    # ------------------------------------------------------------------
    # 0. Resolve cluster_col (handle deprecated query_label_col alias)
    # ------------------------------------------------------------------
    if query_label_col is not None and cluster_col is None:
        warnings.warn(
            "query_label_col is deprecated and will be removed in a future version. "
            "Use cluster_col instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        cluster_col = query_label_col

    if cluster_col is None:
        print(
            "[ZMAP] Note: no cluster_col supplied. Cluster-level consensus aggregation "
            "will be skipped. Pass cluster_col='leiden' (or similar) for a complete summary."
        )
    elif cluster_col not in adata_query.obs.columns:
        print(
            f"[ZMAP] Warning: cluster_col '{cluster_col}' not found in adata_query.obs. "
            "Cluster aggregation will be skipped."
        )
        cluster_col = None

    # ------------------------------------------------------------------
    # 1. Load reference if needed
    # ------------------------------------------------------------------
    if adata_ref is None:
        print(f"[ZMAP] Loading reference ({ref_kind})...")
        from zmap.reference import load_zmap_h5ad
        adata_ref = load_zmap_h5ad(kind=ref_kind)
        print("[ZMAP] Reference loaded.")

    # Effective label namespace
    space = label_space or ref_label_col

    # ------------------------------------------------------------------
    # 2. Preprocess query (TPM + log1p)
    # ------------------------------------------------------------------
    if do_preprocess:
        print("[ZMAP] Preprocessing query — TPM normalization + log1p ...")
        pp_kwargs = dict(preprocess_kwargs or {})
        preprocess_adata_query(
            adata_query,
            counts_source=query_raw_counts_source,
            **pp_kwargs,
        )
        print("[ZMAP] Preprocessing complete.")

    # ------------------------------------------------------------------
    # 3. Symphony mapping / UMAP ingest
    # ------------------------------------------------------------------
    if do_map_embedding:
        print("[ZMAP] Mapping query to ZMAP Symphony embedding...")

        try:
            import symphonypy as sp
        except ImportError as e:
            raise ImportError(
                "Symphony (`import symphonypy as sp`) is required for mapping. "
                "Install symphonypy or disable do_map_embedding."
            ) from e

        sp.tl.map_embedding(adata_query, adata_ref)
        print("[ZMAP] Mapping complete.")

        if do_ingest:
            print("[ZMAP] Ingesting query into reference UMAP...")
            sp.tl.ingest(adata_query=adata_query, adata_ref=adata_ref)
            print("[ZMAP] Ingestion complete.")

    # ------------------------------------------------------------------
    # 4. kNN label transfer
    # ------------------------------------------------------------------
    print("[ZMAP] Running kNN-based label transfer...")
    pk = dict(predict_kwargs or {})
    pk.setdefault("ref_basis", "X_pca_harmony")
    pk.setdefault("query_basis", "X_pca_harmony")
    pk.setdefault("metric", "cosine")
    pk.setdefault("label_suffix", "predicted")   # ensures obs columns are always {space}_predicted
    pk.setdefault("output_dir", output_dir)
    use_tissue_aware_knn = bool(
        pk.pop("use_tissue_aware_knn", False) or pk.pop("use_tissue_aware", False)
    )

    if use_tissue_aware_knn:
        print("[ZMAP] Using tissue-aware kNN transfer...")
        predict_label_tissue_kNN(
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
    print("[ZMAP] Label transfer finished.")

    # ------------------------------------------------------------------
    # 5. Run summary (key/value metadata table)
    # ------------------------------------------------------------------
    df_summary = summarize_knn_run(adata_query, space)

    adata_query.uns.setdefault("zmap_labels", {}).setdefault(space, {})
    adata_query.uns["zmap_labels"][space]["Run Summary Simple"] = df_summary

    if print_summary:
        print("\n[ZMAP] ── Run Summary ──────────────────────────────────────")
        try:
            from IPython.display import display
            display(df_summary)
        except Exception:
            print(df_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Per-cell annotation table
    # ------------------------------------------------------------------
    print("[ZMAP] Building per-cell annotation table...")
    df_cells = build_cell_annotations_table(
        adata_query,
        space,
        cluster_col=cluster_col,
        save_csv=save_outputs,
        output_dir=output_dir,
    )
    adata_query.uns["zmap_labels"][space]["Cell Annotations"] = df_cells

    if print_summary:
        n_cells = len(df_cells)
        print(f"\n[ZMAP] ── Cell Annotations ({n_cells:,} cells) ──────────────────")
        try:
            from IPython.display import display
            display(df_cells.head(10))
            if n_cells > 10:
                print(f"       … {n_cells - 10:,} additional rows (full table in uns and CSV)")
        except Exception:
            print(df_cells.head(10).to_string(index=False))
            if n_cells > 10:
                print(f"       … {n_cells - 10:,} additional rows")

    # ------------------------------------------------------------------
    # 7. Cluster-level consensus aggregation
    # ------------------------------------------------------------------
    if cluster_col is not None:
        print(f"[ZMAP] Aggregating cell annotations by cluster ('{cluster_col}')...")
        try:
            df_clusters = aggregate_by_cluster(
                adata_query,
                cluster_col=cluster_col,
                label_space=space,
                save_csv=save_outputs,
                output_dir=output_dir,
            )
            adata_query.uns["zmap_labels"][space]["Cluster Summary"] = df_clusters

            if print_summary:
                n_clusters = len(df_clusters)
                print(f"\n[ZMAP] ── Cluster Summary ({n_clusters} clusters) ─────────────────")
                try:
                    from IPython.display import display
                    display(df_clusters)
                except Exception:
                    print(df_clusters.to_string(index=False))

            print("[ZMAP] Cluster aggregation complete.")
        except Exception as e:
            print(f"[ZMAP] Warning: cluster aggregation failed: {e}")

    # ------------------------------------------------------------------
    # 8. UMAP overlay figure
    # ------------------------------------------------------------------
    adata_query.uns['ZMAP_CellType_colors'] = adata_ref.uns['ZMAP_CellType_colors'].copy()
    adata_query.uns['ZMAP_Tissue_colors']   = adata_ref.uns['ZMAP_Tissue_colors'].copy()
    adata_query.uns['ZMAP_GermLayer_colors'] = adata_ref.uns['ZMAP_GermLayer_colors'].copy()
    try:
        print("[ZMAP] Plotting UMAP overlay with predicted labels...")
        plot_embedding_with_ondata_labels(
            adata_ref,
            adata_query,
            color_key=f"{space}_predicted",
            show=show_plots,
            save=save_outputs,
            output_dir=output_dir,
        )
        print("[ZMAP] UMAP overlay figure saved.")
    except Exception as e:
        print(f"[ZMAP] Warning: failed to generate UMAP overlay figure: {e}")

    # ------------------------------------------------------------------
    # 9. Label overlap heatmap (cluster × ZMAP label)
    # ------------------------------------------------------------------
    if cluster_col is not None:
        try:
            print(
                f"[ZMAP] Computing label overlap: "
                f"'{cluster_col}' (rows) vs '{ref_label_col}' (columns)..."
            )
            mapping_df = map_query_labels(
                adata_query,
                obs_A=ref_label_col,
                obs_B=cluster_col,
                normalize="row",
                show_plot=show_plots,
                return_df=True,
                save_plots=save_outputs,
                save_mapping=False,
                file_prefix=space,
                output_dir=output_dir,
            )
            adata_query.uns["zmap_labels"][space]["Label Mapping"] = mapping_df
            print("[ZMAP] Label overlap mapping complete.")
        except Exception as e:
            print(f"[ZMAP] Warning: failed to compute label mapping: {e}")

    print(f"\n[ZMAP] ✓ Annotation complete. Results stored under namespace '{space}'.")
    print(f"[ZMAP]   Access summary:  adata.uns['zmap_labels']['{space}']")
    print(f"[ZMAP]   Cell table:      adata.obs['{space}_predicted'], '{space}_prob', ...")
    if cluster_col is not None:
        print(f"[ZMAP]   Cluster table:   adata.uns['zmap_labels']['{space}']['Cluster Summary']")
    return adata_query
