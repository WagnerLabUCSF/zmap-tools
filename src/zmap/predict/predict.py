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

from sklearn.neighbors import NearestNeighbors
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
    Prepare adata_query for ZMAP/Symphony:

      - Requires explicit location of raw counts.
      - Ignores other layers.
      - Performs library-size normalization to `target_sum`.
      - Applies log1p.
      - Writes the result into adata.X.
      - Records metadata in adata.uns['ZMAP_preprocessing'].
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
):
    """
    kNN label transfer with pre-masking of reference cells prior to kNN construction.

    Key change vs. earlier versions:
    - We filter the reference (omit_labels / NaNs) BEFORE building the kNN index.
      This keeps the voting denominator at exactly k and restores clean 1/k probability steps.
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
    if 'zmap_neighbors' in adata_query.uns:
        cache = adata_query.uns['zmap_neighbors']
        same_config = (
            cache.get('ref_basis') == ref_basis and
            cache.get('query_basis') == query_basis and
            cache.get('metric') == metric and
            cache.get('n_neighbors') == n_neighbors and
            cache.get('mask_digest') == _mask_digest
        )
        if same_config:
            print("Reusing cached neighbor graph from adata_query.uns['zmap_neighbors'] (filtered).")
            neighbor_indices = cache['indices']
            distances = cache['distances']
            reuse_neighbors = True

    if not reuse_neighbors:
        print("Computing new kNN graph on filtered reference...")
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(X_ref)
        distances, neighbor_indices = nn.kneighbors(adata_query.obsm[query_basis], return_distance=True)
        adata_query.uns['zmap_neighbors'] = {
            'indices': neighbor_indices,
            'distances': distances,
            'ref_basis': ref_basis,
            'query_basis': query_basis,
            'metric': metric,
            'n_neighbors': n_neighbors,
            'mask_digest': _mask_digest,
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

            # directory: working_directory/zmap/qc
            qc_dir = os.path.join(os.getcwd(), "zmap", "qc")
            os.makedirs(qc_dir, exist_ok=True)

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
                prob_path = os.path.join(qc_dir, "probability_qc.png")
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
                dist_path = os.path.join(qc_dir, "distance_qc.png")
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
    Backward-compatible summary of prediction run stored in:
        adata_query.uns['zmap_labels'][label_key]['Run Summary']
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
        ("p_thresh", params.get("p_thresh")),
        ("n_assigned", cov.get("n_assigned")),
        ("pct_assigned", cov.get("pct_assigned")),
    ], columns=["Key", "Value"])

    return df


# ================================================================
#  3. Horizontal histogram (time distribution bar)
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
    Horizontal histogram strip used for time distribution visualization.
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
    Synchronize categorical color palettes across query and reference AnnData.
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
):
    """
    Plot reference embedding + query overlay with on-data labels, synced ZMAP colors,
    and (optionally) a rotated vertical time distribution strip on the RIGHT
    built from ZMAP_time_id.

    - Uses sync_zmap_colors to harmonize palettes between ref/query.
    - Uses plot_colorbar_histogram(adata_test.obs[time_key]) drawn into a temporary
      axis, then transposed and re-plotted as a vertical strip on the right.
    """
    # ---- prepare test AnnData (drop NAs on requested key, cast to categorical) ----
    if filter_na:
        mask = ~adata_test.obs[color_key].isna()
        adata_test_plot = adata_test[mask].copy()
    else:
        adata_test_plot = adata_test.copy()

    adata_test_plot.obs[color_key] = adata_test_plot.obs[color_key].astype("category")

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

        # --- recolor existing on-data text labels by palette ---
        texts = [t for t in ax.texts]
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
        adjust_text(
            texts,
            ax=ax,
            expand=adjust_expand,
            arrowprops=arrowprops,
            min_arrow_len=min_arrow_len,
        )

        # --- match arrow color to label color ---
        if match_arrow_color_to_text:
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
            outdir = os.path.join(os.getcwd(), "zmap", "predict")
            os.makedirs(outdir, exist_ok=True)

            safe_name = color_key.replace("/", "_").replace(" ", "_")
            png_path = os.path.join(outdir, f"{safe_name}.png")
            pdf_path = os.path.join(outdir, f"{safe_name}.pdf")

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
):
    """
    Compare two labelings in an AnnData object's .obs and compute an overlap matrix.

    Parameters
    ----------
    adata_query : AnnData
        The AnnData object containing the two label columns.
    obs_A : str
        Column name in adata_query.obs to use as labeling A (columns of matrix).
    obs_B : str
        Column name in adata_query.obs to use as labeling B (rows of matrix).

    Saving behavior
    ---------------
    - If save_plots=True → saves PNG + PDF to ./zmap/predict/
    - If save_mapping=True → saves CSV mapping table to ./zmap/predict/

    Returns
    -------
    mapping_df or None
        Cluster-level best match table for obs_B → obs_A.
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
    save_dir = os.path.join("zmap", "predict")
    os.makedirs(save_dir, exist_ok=True)

    if save_plots and fig is not None:
        base = f"ZMAP_overlap_{obs_B}_vs_{obs_A}"
        fig.savefig(os.path.join(save_dir, f"{base}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(save_dir, f"{base}.pdf"), bbox_inches="tight")
        print(f"[ZMAP] Saved overlap figure → {save_dir}")

    if save_mapping and mapping_df is not None:
        out_csv = os.path.join(save_dir, f"ZMAP_label_mapping_{obs_B}_vs_{obs_A}.csv")
        mapping_df.to_csv(out_csv)
        print(f"[ZMAP] Saved mapping table → {out_csv}")

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
    query_label_col: str | None = None,   # e.g. "leiden" for overlap mapping

    # --- pipeline toggles ---
    do_preprocess: bool = True,
    do_map_embedding: bool = True,
    do_ingest: bool = True,

    # --- kwargs passthroughs to lower-level steps ---
    preprocess_kwargs: Mapping[str, Any] | None = None,
    predict_kwargs: Mapping[str, Any] | None = None,

    # --- output controls ---
    print_summary: bool = True,
) -> ad.AnnData:
    """
    High-level ZMAP/Symphony prediction wrapper.

    Steps
    -----
    1. (Optional) Preprocess query from raw counts (TPM + log1p).
    2. (Optional) Symphony mapping + ingest.
    3. kNN label transfer with predict_labels_kNN.
    4. Store simple run summary in:
         adata_query.uns['zmap_labels'][<space>]['Run Summary Simple']
    5. Plot UMAP overlay with on-data labels:
         plot_embedding_with_ondata_labels(adata_ref, adata_query,
                                           color_key=ref_label_col)
    6. (Optional) Map query labels (e.g. 'leiden') to ZMAP labels via:
         map_labels(adata_query, obs_A=ref_label_col, obs_B=query_label_col)
       and store mapping in:
         adata_query.uns['zmap_labels'][<space>]['Label Mapping']
    """

    # Suppress UMAP "n_jobs overridden" warnings
    warnings.filterwarnings(
        "ignore",
        message=".*overridden to 1 by setting random_state.*"
    )

    # -----------------------------
    # 0. Load reference if needed
    # -----------------------------
    if adata_ref is None:
        print(f"[ZMAP] Loading reference ({ref_kind})...")
        from zmap.ref import load_zmap_h5ad
        adata_ref = load_zmap_h5ad(kind=ref_kind)
        print("[ZMAP] Reference loaded.")

    # Effective label namespace
    space = label_space or ref_label_col

    # -----------------------------
    # 1. Preprocess query (TPM+log1p)
    # -----------------------------
    if do_preprocess:
        print("[ZMAP] Preprocessing query — TPM normalization + log1p ...")
        pp_kwargs = dict(preprocess_kwargs or {})
        preprocess_adata_query(
            adata_query,
            counts_source=query_raw_counts_source,
            **pp_kwargs,
        )
        print("[ZMAP] Preprocessing complete.")

    # -----------------------------
    # 2. Symphony mapping / ingest
    # -----------------------------
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

    # -----------------------------
    # 3. kNN label transfer
    # -----------------------------
    print("[ZMAP] Running kNN-based label transfer...")
    pk = dict(predict_kwargs or {})
    pk.setdefault("ref_basis", "X_pca_harmony")
    pk.setdefault("query_basis", "X_pca_harmony")
    pk.setdefault("metric", "cosine")

    predict_labels_kNN(
        adata_query,
        adata_ref,
        ref_label_col=ref_label_col,
        label_space=space,
        query_truth_col=query_truth_col,
        **pk,
    )
    print("[ZMAP] Label transfer finished.")

    # -----------------------------
    # 4. Build + store simplified summary
    # -----------------------------
    df_summary = summarize_knn_run(adata_query, space)

    adata_query.uns.setdefault("zmap_labels", {}).setdefault(space, {})
    adata_query.uns["zmap_labels"][space]["Run Summary Simple"] = df_summary

    if print_summary:
        try:
            from IPython.display import display
            display(adata_query.uns["zmap_labels"][space]["Run Summary Simple"])
        except Exception:
            # fall back to plain print if display isn't available
            print(adata_query.uns["zmap_labels"][space]["Run Summary Simple"])

    # -----------------------------
    # 5. UMAP overlay figure
    # -----------------------------
    try:
        print("[ZMAP] Plotting UMAP overlay with predicted labels...")
        plot_embedding_with_ondata_labels(
            adata_ref,
            adata_query,
            color_key=ref_label_col,  # use the main label column in obs
            show=False,               # save to disk, don't pop up interactively
            save=True,                # default in that function, but explicit here
        )
        print("[ZMAP] UMAP overlay figure saved.")
    except Exception as e:
        print(f"[ZMAP] Warning: failed to generate UMAP overlay figure: {e}")

    # -----------------------------
    # 6. Label overlap mapping (query_label_col vs. ref_label_col)
    # -----------------------------
    if query_label_col is not None:
        if query_label_col not in adata_query.obs.columns:
            print(
                f"[ZMAP] Warning: query_label_col '{query_label_col}' not found in "
                f"adata_query.obs; skipping label mapping."
            )
        else:
            try:
                print(
                    f"[ZMAP] Computing label overlap: "
                    f"'{query_label_col}' (rows) vs '{ref_label_col}' (columns)..."
                )
                mapping_df = map_query_labels(
                    adata_query,
                    obs_A=ref_label_col,
                    obs_B=query_label_col,
                    normalize="row",
                    show_plot=True,    # heatmap, also saved by map_labels
                    return_df=True,
                )
                # store mapping in uns
                adata_query.uns["zmap_labels"][space]["Label Mapping"] = mapping_df
                print("[ZMAP] Label mapping complete and stored in adata_query.uns.")
            except Exception as e:
                print(f"[ZMAP] Warning: failed to compute label mapping: {e}")

    print(f"[ZMAP] Annotation complete. Results stored under namespace '{space}'.")
    return adata_query

