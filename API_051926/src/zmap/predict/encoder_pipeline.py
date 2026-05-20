from __future__ import annotations

import json
import os
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .knn_backend import knn_search
from ..reference import load_zmap_h5ad

# Change it to sys later
DEFAULT_ENCODER_MODEL_SUBDIR = Path(
    "models/phase3_trim_normalize/model"
)
DEFAULT_UMAP_REDUCER_SUBPATH = Path(
    "models/phase3_trim_normalize/umap_reducer/umap_reducer.joblib"
)
DEFAULT_REF_H5AD_FILENAME = "ZMAP_251209_processed_super_slim.h5ad"
DEFAULT_REF_H5AD_FILENAME_FALLBACK = "ZMAP_251209_processed_slim.h5ad"
DEFAULT_REF_H5AD_FILENAME_FALLBACK2 = "ZMAP_251209_processed.h5ad"
DEFAULT_MODULE_MARKERS_CSV_FILENAME = "all_levels_selected_markers.csv"

# Legacy local paths (kept only as backward-compatible fallback).
LEGACY_ENCODER_MODEL_DIR = Path(
    "/c4/home/yingxins/ZMAP/output/phase3_trim_normalize/model"
)
LEGACY_UMAP_REDUCER_PATH = Path(
    "/c4/home/yingxins/ZMAP/output/phase3_trim_normalize/umap_reducer/umap_reducer.joblib"
)
LEGACY_UMAP_REDUCER_PATH_ALT = Path(
    "/c4/home/yingxins/ZMAP/output/umap_regen_251209_xpca/umap_reducer.joblib"
)
LEGACY_REF_H5AD_PATH = Path("/c4/home/yingxins/ZMAP/data/ZMAP_251209_processed_super_slim.h5ad")
LEGACY_REF_H5AD_PATH_FALLBACK = Path("/c4/home/yingxins/ZMAP/data/ZMAP_251209_processed_slim.h5ad")
LEGACY_REF_H5AD_PATH_FALLBACK2 = Path("/c4/home/yingxins/ZMAP/data/ZMAP_251209_processed.h5ad")
LEGACY_MODULE_MARKERS_CSV_PATH = Path(
    "/c4/home/yingxins/ZMAP/output/"
    "module_gene_lists_v3_top20_support0_no_germlayer_no_level_overlap_lca_reassign_"
    "drop_shared_keep_orig_tissue_no_unassigned/all_levels_selected_markers.csv"
)

DEFAULT_REF_LATENT_KEY = "X_pca"
DEFAULT_QUERY_LATENT_KEY = "X_pca_pred"
DEFAULT_LABEL_COL = "ZMAP_CellType"
DEFAULT_TISSUE_COL = "ZMAP_Tissue"
DEFAULT_REF_KIND_FALLBACK = "processed"
_PINNED_REF_CACHE: ad.AnnData | None = None
_PINNED_REF_SOURCE: str | None = None


def _resolve_tissue_aware_mode(
    *,
    tissue_aware: bool | None,
    tissue_mode: str,
) -> tuple[bool, str]:
    if tissue_mode not in {"none", "hard", "soft"}:
        raise ValueError("tissue_mode must be one of {'none', 'hard', 'soft'}.")
    if tissue_aware is None:
        enabled = tissue_mode != "none"
    else:
        enabled = bool(tissue_aware)
    if not enabled:
        return False, "none"
    if tissue_mode == "none":
        raise ValueError("tissue_aware=True requires tissue_mode='hard' or tissue_mode='soft'.")
    return True, str(tissue_mode)


def knn_config(
    *,
    ref_latent_key: str = DEFAULT_REF_LATENT_KEY,
    query_latent_key: str = "X_pca_pred",
    k: int = 15,
    tissue_aware: bool = False,
    tissue_mode: str = "hard",
    tissue_col: str | None = None,
    predict_tissue_level: bool = False,
    tissue_penalty_lambda: float = 1.0,
    knn_metric: str = "euclidean",
    knn_backend: str = "auto",
    knn_device: str = "auto",
    knn_nprobe: int | None = None,
    knn_l2norm: bool = False,
    class_prior_alpha: float = 0.0,
    pseudo_tissue_k: int | None = None,
    pseudo_tissue_threshold: float = 0.0,
    pseudo_tissue_margin_threshold: float = 0.0,
    reuse_knn_cache: bool = True,
    include_unassigned: bool = False,
) -> dict[str, object]:
    """
    Build a shared KNN config dict for encoder-pipeline utilities.
    """
    if int(k) <= 0:
        raise ValueError("k must be positive.")
    if tissue_mode not in {"none", "hard", "soft"}:
        raise ValueError("tissue_mode must be one of {'none', 'hard', 'soft'}.")
    tissue_aware_resolved, tissue_mode_resolved = _resolve_tissue_aware_mode(
        tissue_aware=bool(tissue_aware),
        tissue_mode=tissue_mode,
    )
    if tissue_aware_resolved and tissue_col is None and not bool(predict_tissue_level):
        raise ValueError(
            "tissue_aware=True requires tissue_col, or set predict_tissue_level=True "
            "to infer query tissue labels before tissue-aware kNN."
        )
    if knn_metric not in {"euclidean", "cosine"}:
        raise ValueError("knn_metric must be one of {'euclidean', 'cosine'}.")
    if knn_backend not in {"auto", "faiss", "sklearn"}:
        raise ValueError("knn_backend must be one of {'auto', 'faiss', 'sklearn'}.")
    if knn_nprobe is not None and int(knn_nprobe) <= 0:
        raise ValueError("knn_nprobe must be positive when provided.")
    if pseudo_tissue_k is not None and int(pseudo_tissue_k) <= 0:
        raise ValueError("pseudo_tissue_k must be positive when provided.")

    return {
        "ref_latent_key": str(ref_latent_key),
        "query_latent_key": str(query_latent_key),
        "k": int(k),
        "tissue_aware": bool(tissue_aware_resolved),
        "tissue_mode": str(tissue_mode_resolved),
        "tissue_col": (None if tissue_col is None else str(tissue_col)),
        "predict_tissue_level": bool(predict_tissue_level),
        "tissue_penalty_lambda": float(tissue_penalty_lambda),
        "knn_metric": str(knn_metric),
        "knn_backend": str(knn_backend),
        "knn_device": str(knn_device),
        "knn_nprobe": (None if knn_nprobe is None else int(knn_nprobe)),
        "knn_l2norm": bool(knn_l2norm),
        "class_prior_alpha": float(class_prior_alpha),
        "pseudo_tissue_k": (None if pseudo_tissue_k is None else int(pseudo_tissue_k)),
        "pseudo_tissue_threshold": float(pseudo_tissue_threshold),
        "pseudo_tissue_margin_threshold": float(pseudo_tissue_margin_threshold),
        "reuse_knn_cache": bool(reuse_knn_cache),
        "include_unassigned": bool(include_unassigned),
    }


def _infer_time_order(categories: list[str]) -> list[str]:
    def _extract_num(label: str) -> float | None:
        m = re.search(r"[-+]?\d*\.?\d+", label)
        return float(m.group()) if m else None

    nums = [_extract_num(c) for c in categories]
    if all(n is not None for n in nums):
        return [c for _, c in sorted(zip(nums, categories), key=lambda x: (x[0], x[1]))]
    return sorted(categories)


def _time_regression_from_neighbors(
    ref_time: np.ndarray,
    idx: np.ndarray,
    dist: np.ndarray,
    *,
    tau: float,
    n_classes: int,
    topk: int,
    query_time_enc: np.ndarray | None,
    monotone_delta: int,
    monotone_gamma: float,
    trim_extremes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_query = idx.shape[0]
    time_pred = np.full(n_query, np.nan, dtype=np.float32)
    time_var = np.full(n_query, np.nan, dtype=np.float32)
    mean_dist = np.full(n_query, np.nan, dtype=np.float32)
    time_prob = np.zeros((n_query, n_classes), dtype=np.float32)

    for i in range(n_query):
        row_idx = idx[i]
        valid = row_idx >= 0
        if not np.any(valid):
            continue
        row_idx = row_idx[valid]
        row_dist = dist[i][valid]

        tau_i = np.median(row_dist) if tau <= 0 else max(float(tau), 1e-8)
        weights = np.exp(-row_dist / max(tau_i, 1e-8))
        denom = float(weights.sum())
        if denom <= 0:
            continue
        weights = weights / denom

        t_vals = ref_time[row_idx]
        valid_time = np.isfinite(t_vals)
        if not np.any(valid_time):
            continue
        weights = weights * valid_time
        denom = float(weights.sum())
        if denom <= 0:
            continue
        weights = weights / denom

        if (
            query_time_enc is not None
            and monotone_delta > 0
            and monotone_gamma < 1.0
            and np.isfinite(query_time_enc[i])
        ):
            cutoff = query_time_enc[i] - float(monotone_delta)
            mask = t_vals <= cutoff
            if np.any(mask):
                weights = weights.copy()
                weights[mask] *= float(monotone_gamma)
                denom = float(weights.sum())
                if denom <= 0:
                    continue
                weights = weights / denom

        if topk > 0:
            keep = min(int(topk), weights.size)
            top_idx = np.argsort(weights)[-keep:]
            mask = np.zeros_like(weights, dtype=bool)
            mask[top_idx] = True
            weights = np.where(mask, weights, 0.0)
            denom = float(weights.sum())
            if denom <= 0:
                continue
            weights = weights / denom

        if trim_extremes > 0:
            keep = weights > 0
            if int(np.sum(keep)) > int(trim_extremes) * 2:
                t_vals_trim = t_vals[keep]
                w_trim = weights[keep]
                order_idx = np.argsort(t_vals_trim)
                trim = min(int(trim_extremes), (len(order_idx) - 1) // 2)
                if trim > 0:
                    drop_idx = list(order_idx[:trim]) + list(order_idx[-trim:])
                    keep_mask = np.ones_like(w_trim, dtype=bool)
                    keep_mask[drop_idx] = False
                    w_trim = w_trim * keep_mask
                    denom = float(w_trim.sum())
                    if denom <= 0:
                        continue
                    w_trim = w_trim / denom
                    weights = np.zeros_like(weights)
                    weights[np.where(keep)[0]] = w_trim

        t_vals = t_vals[valid_time]
        w_vals = weights[valid_time]
        pred = float((w_vals * t_vals).sum())
        var = float((w_vals * (t_vals - pred) ** 2).sum())
        time_pred[i] = pred
        time_var[i] = var
        mean_dist[i] = float(np.mean(row_dist))

        for t, w in zip(t_vals, w_vals):
            t_int = int(t)
            if 0 <= t_int < n_classes:
                time_prob[i, t_int] += float(w)

    return time_pred, time_var, mean_dist, time_prob


def _time_label_vote_from_neighbors(
    *,
    ref_time_ord: np.ndarray,
    ref_time_label: np.ndarray,
    idx: np.ndarray,
    dist: np.ndarray,
    time_map: dict[str, int],
    hard_topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_query = idx.shape[0]
    pred_label = np.full(n_query, None, dtype=object)
    pred_ord = np.full(n_query, np.nan, dtype=np.float32)
    for i in range(n_query):
        row_idx = idx[i]
        valid = row_idx >= 0
        if not np.any(valid):
            continue
        row_idx = row_idx[valid]
        row_dist = dist[i][valid]
        row_ord = ref_time_ord[row_idx]
        valid_time = np.isfinite(row_ord)
        if not np.any(valid_time):
            continue
        row_idx = row_idx[valid_time]
        row_dist = row_dist[valid_time]
        row_labels = ref_time_label[row_idx]
        if hard_topk > 0:
            keep = min(int(hard_topk), row_idx.size)
            order_idx = np.argsort(row_dist)[:keep]
            row_dist = row_dist[order_idx]
            row_labels = row_labels[order_idx]
        counts: dict[str, int] = {}
        dist_sums: dict[str, float] = {}
        for lbl, d in zip(row_labels, row_dist):
            if lbl not in time_map:
                continue
            counts[lbl] = counts.get(lbl, 0) + 1
            dist_sums[lbl] = dist_sums.get(lbl, 0.0) + float(d)
        if not counts:
            continue
        best = sorted(
            counts.keys(),
            key=lambda x: (-counts[x], dist_sums.get(x, 0.0) / counts[x], time_map.get(x, 0)),
        )[0]
        pred_label[i] = best
        pred_ord[i] = float(time_map[best])
    return pred_label, pred_ord


def _compute_time_prediction_from_knn(
    adata_query: ad.AnnData,
    adata_ref: ad.AnnData,
    *,
    keep: np.ndarray,
    idx: np.ndarray,
    dist: np.ndarray,
    time_col: str,
    time_order: str | list[str] | None,
    time_topk: int,
    time_hard_topk: int,
    time_trim_extremes: int,
    time_tau: float,
    time_monotone_delta: int,
    time_monotone_gamma: float,
    output_prefix: str | None = None,
) -> dict[str, object]:
    if time_topk < 0:
        raise ValueError("time_topk must be >= 0")
    if time_hard_topk < 0:
        raise ValueError("time_hard_topk must be >= 0")
    if time_trim_extremes < 0:
        raise ValueError("time_trim_extremes must be >= 0")
    if time_col not in adata_ref.obs:
        raise KeyError(f"Missing time column in adata_ref.obs: {time_col}")

    ref_time_raw = adata_ref.obs[time_col].to_numpy()
    ref_valid = ~pd.isna(ref_time_raw)
    ref_time_labels = [str(x) for x in ref_time_raw[ref_valid] if str(x) != "unassigned"]
    ref_categories = list(pd.unique(np.asarray(ref_time_labels, dtype=object)))
    if not ref_categories:
        raise ValueError(f"No valid reference time labels found in '{time_col}'.")

    if time_order is None:
        order = _infer_time_order(ref_categories)
    else:
        if isinstance(time_order, str):
            order = [t.strip() for t in time_order.split(",") if t.strip()]
        else:
            order = [str(t).strip() for t in time_order if str(t).strip()]
        missing = set(ref_categories) - set(order)
        if missing:
            raise ValueError(f"time_order missing categories: {sorted(missing)}")

    time_map = {label: i for i, label in enumerate(order)}
    ref_time_enc = np.full(ref_time_raw.shape[0], np.nan, dtype=np.float32)
    for ridx in np.flatnonzero(ref_valid):
        label = str(ref_time_raw[ridx])
        if label != "unassigned" and label in time_map:
            ref_time_enc[ridx] = float(time_map[label])

    ref_time_knn = ref_time_enc[keep]
    ref_time_label_knn = adata_ref.obs[time_col].astype(str).to_numpy()[keep]
    if int(np.isfinite(ref_time_knn).sum()) == 0:
        raise ValueError(f"No valid reference time values after filtering in '{time_col}'.")

    query_time_raw = None
    query_time_enc = None
    if time_col in adata_query.obs:
        query_time_raw = adata_query.obs[time_col].to_numpy()
        query_time_enc = np.full(query_time_raw.shape[0], np.nan, dtype=np.float32)
        q_valid = ~pd.isna(query_time_raw)
        for qidx, label in zip(np.flatnonzero(q_valid), [str(x) for x in query_time_raw[q_valid]]):
            if label != "unassigned" and label in time_map:
                query_time_enc[qidx] = float(time_map[label])

    time_pred, time_var, time_mean_dist, time_prob = _time_regression_from_neighbors(
        ref_time_knn,
        idx,
        dist,
        tau=float(time_tau),
        n_classes=len(order),
        topk=int(time_topk),
        query_time_enc=query_time_enc,
        monotone_delta=int(time_monotone_delta),
        monotone_gamma=float(time_monotone_gamma),
        trim_extremes=int(time_trim_extremes),
    )

    time_entropy = np.full(idx.shape[0], np.nan, dtype=np.float32)
    prob_sum = np.sum(time_prob, axis=1)
    valid_prob = prob_sum > 0
    if np.any(valid_prob):
        p = time_prob[valid_prob] / prob_sum[valid_prob][:, None]
        time_entropy[valid_prob] = -np.sum(p * np.log(p + 1e-12), axis=1)
        finite_entropy = np.isfinite(time_entropy)
        if np.any(finite_entropy):
            entropy_p95 = float(np.nanquantile(time_entropy[finite_entropy], 0.95))
            high_entropy = np.isfinite(time_entropy) & (time_entropy >= entropy_p95)
            time_pred[high_entropy] = np.nan
            time_var[high_entropy] = np.nan
            time_prob[high_entropy] = 0.0

    time_pred_label, time_pred_ordinal = _time_label_vote_from_neighbors(
        ref_time_ord=ref_time_knn,
        ref_time_label=ref_time_label_knn,
        idx=idx,
        dist=dist,
        time_map=time_map,
        hard_topk=int(time_hard_topk),
    )

    delay = np.full(idx.shape[0], np.nan, dtype=np.float32)
    eval_summary: dict[str, float | int | None] = {
        "n_valid": 0,
        "exact_accuracy": None,
        "ordinal_accuracy_pm1": None,
        "mean_abs_error": None,
    }
    confusion = None
    if query_time_enc is not None:
        valid_time = np.isfinite(query_time_enc)
        delay[valid_time] = time_pred[valid_time] - query_time_enc[valid_time]
        valid_acc = valid_time & np.isfinite(time_pred_ordinal)
        if np.any(valid_acc):
            diff = np.abs(time_pred_ordinal[valid_acc] - query_time_enc[valid_acc])
            eval_summary = {
                "n_valid": int(np.sum(valid_acc)),
                "exact_accuracy": float(np.mean(diff == 0)),
                "ordinal_accuracy_pm1": float(np.mean(diff <= 1)),
                "mean_abs_error": float(np.mean(diff)),
            }
            y_true = np.array([str(x) for x in query_time_raw[valid_acc]], dtype=object)
            y_pred = time_pred_label[valid_acc].astype(str)
            cm = pd.crosstab(
                pd.Categorical(y_true, categories=order),
                pd.Categorical(y_pred, categories=order),
                dropna=False,
            )
            confusion = {
                "labels": order,
                "matrix": cm.to_numpy(dtype=int).tolist(),
            }

    prefix = output_prefix or str(time_col)
    prob_key = f"{prefix}_time_probabilities"
    adata_query.obs[f"{prefix}_pred"] = time_pred
    adata_query.obs[f"{prefix}_var"] = time_var
    adata_query.obs[f"{prefix}_delay"] = delay
    adata_query.obs[f"{prefix}_pred_label"] = time_pred_label
    adata_query.obs[f"{prefix}_pred_ordinal"] = time_pred_ordinal
    adata_query.obs[f"{prefix}_knn_mean_dist"] = time_mean_dist
    adata_query.obs[f"{prefix}_entropy"] = time_entropy
    if query_time_enc is not None:
        adata_query.obs[f"{prefix}_obs_ordinal"] = query_time_enc
    adata_query.obsm[prob_key] = time_prob.astype(np.float32, copy=False)

    out = {
        "time_col": str(time_col),
        "time_order": order,
        "time_map": {k: int(v) for k, v in time_map.items()},
        "time_topk": int(time_topk),
        "time_hard_topk": int(time_hard_topk),
        "time_trim_extremes": int(time_trim_extremes),
        "time_tau": float(time_tau),
        "time_monotone_delta": int(time_monotone_delta),
        "time_monotone_gamma": float(time_monotone_gamma),
        "prob_key": prob_key,
        "output_prefix": prefix,
        "evaluation": eval_summary,
    }
    if confusion is not None:
        out["confusion_matrix"] = confusion
    return out


def _zmap_home_dir() -> Path:
    env_home = os.environ.get("ZMAP_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser()
    xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg_data_home:
        return Path(xdg_data_home).expanduser() / "zmap"
    return Path.home() / ".local" / "share" / "zmap"


def _resolve_encoder_model_dir() -> tuple[Path, str]:
    """
    Resolve encoder model directory with priority:
      1) ZMAP_ENCODER_MODEL_DIR
      2) <zmap_home>/models/... (package default)
      3) legacy absolute path (backward compatibility)
    """
    env_model = os.environ.get("ZMAP_ENCODER_MODEL_DIR", "").strip()
    candidates: list[tuple[Path, str]] = []
    if env_model:
        candidates.append((Path(env_model).expanduser(), "env:ZMAP_ENCODER_MODEL_DIR"))
    candidates.append((_zmap_home_dir() / DEFAULT_ENCODER_MODEL_SUBDIR, "zmap_home_default"))
    candidates.append((LEGACY_ENCODER_MODEL_DIR, "legacy_local_fallback"))

    for model_dir, source in candidates:
        cfg = model_dir / "config.json"
        ckpt = model_dir / "encoder.pt"
        if cfg.exists() and ckpt.exists():
            return model_dir, source

    tried = "\n".join([f"- {str(p)} ({src})" for p, src in candidates])
    raise FileNotFoundError(
        "Could not find encoder model files (config.json + encoder.pt).\n"
        "Set ZMAP_ENCODER_MODEL_DIR, or place model under:\n"
        f"- {str(_zmap_home_dir() / DEFAULT_ENCODER_MODEL_SUBDIR)}\n"
        "Tried:\n"
        f"{tried}"
    )


def _resolve_umap_reducer_path(umap_reducer_path: str | None = None) -> tuple[Path, str]:
    """
    Resolve UMAP reducer path with priority:
      1) `umap_reducer_path` argument
      2) `ZMAP_UMAP_REDUCER`
      3) `<zmap_home>/models/...` package default
      4) legacy absolute path
    """
    candidates: list[tuple[Path, str]] = []
    if umap_reducer_path is not None and str(umap_reducer_path).strip():
        candidates.append((Path(str(umap_reducer_path)).expanduser(), "arg:umap_reducer_path"))
    env_path = os.environ.get("ZMAP_UMAP_REDUCER", "").strip()
    if env_path:
        candidates.append((Path(env_path).expanduser(), "env:ZMAP_UMAP_REDUCER"))
    candidates.append((_zmap_home_dir() / DEFAULT_UMAP_REDUCER_SUBPATH, "zmap_home_default"))
    candidates.append((LEGACY_UMAP_REDUCER_PATH_ALT, "legacy_local_alt_fallback"))
    candidates.append((LEGACY_UMAP_REDUCER_PATH, "legacy_local_fallback"))

    for p, src in candidates:
        if p.exists() and p.is_file():
            return p, src

    tried = "\n".join([f"- {str(p)} ({src})" for p, src in candidates])
    raise FileNotFoundError(
        "Could not find UMAP reducer joblib.\n"
        "Set `umap_reducer_path`, env `ZMAP_UMAP_REDUCER`, or place it under:\n"
        f"- {str(_zmap_home_dir() / DEFAULT_UMAP_REDUCER_SUBPATH)}\n"
        "Tried:\n"
        f"{tried}"
    )


def _resolve_ref_h5ad_path() -> tuple[Path | None, str]:
    """
    Resolve reference h5ad path with priority:
      1) ZMAP_REF_H5AD
      2) <zmap_home>/h5ads/ZMAP_251209_processed_super_slim.h5ad
      3) <zmap_home>/h5ads/ZMAP_251209_processed_slim.h5ad
      4) <zmap_home>/h5ads/ZMAP_251209_processed.h5ad
      5) legacy absolute super_slim path
      6) legacy absolute slim path
      7) legacy absolute processed path
    """
    env_ref = os.environ.get("ZMAP_REF_H5AD", "").strip()
    candidates: list[tuple[Path, str]] = []
    if env_ref:
        candidates.append((Path(env_ref).expanduser(), "env:ZMAP_REF_H5AD"))
    candidates.append(
        (_zmap_home_dir() / "h5ads" / DEFAULT_REF_H5AD_FILENAME, "zmap_home_default_slim")
    )
    candidates.append(
        (
            _zmap_home_dir() / "h5ads" / DEFAULT_REF_H5AD_FILENAME_FALLBACK,
            "zmap_home_fallback_slim",
        )
    )
    candidates.append(
        (
            _zmap_home_dir() / "h5ads" / DEFAULT_REF_H5AD_FILENAME_FALLBACK2,
            "zmap_home_fallback_processed",
        )
    )
    candidates.append((LEGACY_REF_H5AD_PATH, "legacy_local_default_super_slim"))
    candidates.append((LEGACY_REF_H5AD_PATH_FALLBACK, "legacy_local_fallback_slim"))
    candidates.append((LEGACY_REF_H5AD_PATH_FALLBACK2, "legacy_local_fallback_processed"))

    for ref_path, source in candidates:
        if ref_path.exists():
            return ref_path, source
    return None, "download_fallback"


def _h5ad_has_expression_layer(path: Path, layer_key: str) -> bool:
    try:
        import h5py

        with h5py.File(path, "r") as f:
            if "var" not in f or "_index" not in f["var"]:
                return False
            if int(f["var"]["_index"].shape[0]) <= 0:
                return False
            if layer_key == "X":
                if "X" not in f:
                    return False
                shape = f["X"].shape
                return len(shape) == 2 and int(shape[1]) > 0
            if "layers" not in f or layer_key not in f["layers"]:
                return False
            shape = f["layers"][layer_key].shape
            return len(shape) == 2 and int(shape[1]) > 0
    except Exception:
        return False


def _resolve_expression_ref_h5ad_path(layer_key: str) -> tuple[Path | None, str]:
    env_ref = os.environ.get("ZMAP_REF_H5AD", "").strip()
    candidates: list[tuple[Path, str]] = []
    if env_ref:
        candidates.append((Path(env_ref).expanduser(), "env:ZMAP_REF_H5AD"))
    candidates.append(
        (
            _zmap_home_dir() / "h5ads" / DEFAULT_REF_H5AD_FILENAME_FALLBACK2,
            "zmap_home_fallback_processed",
        )
    )
    candidates.append((LEGACY_REF_H5AD_PATH_FALLBACK2, "legacy_local_fallback_processed"))
    # Try slim/super as additional fallback in case user has custom builds.
    candidates.append(
        (_zmap_home_dir() / "h5ads" / DEFAULT_REF_H5AD_FILENAME_FALLBACK, "zmap_home_fallback_slim")
    )
    candidates.append((LEGACY_REF_H5AD_PATH_FALLBACK, "legacy_local_fallback_slim"))
    candidates.append((_zmap_home_dir() / "h5ads" / DEFAULT_REF_H5AD_FILENAME, "zmap_home_default_super_slim"))
    candidates.append((LEGACY_REF_H5AD_PATH, "legacy_local_default_super_slim"))

    for ref_path, source in candidates:
        if ref_path.exists() and _h5ad_has_expression_layer(ref_path, layer_key):
            return ref_path, source
    return None, "not_found_expression_ref"


def _resolve_module_markers_csv_path(module_csv: str | None = None) -> tuple[Path | None, str]:
    """
    Resolve module markers CSV path with priority:
      1) explicit module_csv argument
      2) ZMAP_MODULE_CSV
      3) <zmap_home>/modules/all_levels_selected_markers.csv
      4) legacy pinned absolute path
    """
    candidates: list[tuple[Path, str]] = []
    if module_csv is not None and str(module_csv).strip() != "":
        candidates.append((Path(str(module_csv)).expanduser(), "arg:module_csv"))
    env_mod = os.environ.get("ZMAP_MODULE_CSV", "").strip()
    if env_mod:
        candidates.append((Path(env_mod).expanduser(), "env:ZMAP_MODULE_CSV"))
    candidates.append(
        (
            _zmap_home_dir() / "modules" / DEFAULT_MODULE_MARKERS_CSV_FILENAME,
            "zmap_home_default_module_csv",
        )
    )
    candidates.append((LEGACY_MODULE_MARKERS_CSV_PATH, "legacy_local_default_module_csv"))

    for mod_path, source in candidates:
        if mod_path.exists():
            return mod_path, source
    return None, "not_found"


def _ensure_model_files(model_dir: Path) -> tuple[Path, Path]:
    cfg_path = model_dir / "config.json"
    ckpt_path = model_dir / "encoder.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing encoder config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing encoder checkpoint: {ckpt_path}")
    return cfg_path, ckpt_path


def _load_pinned_reference(*, use_cache: bool = True) -> tuple[ad.AnnData, str]:
    global _PINNED_REF_CACHE, _PINNED_REF_SOURCE
    if use_cache and _PINNED_REF_CACHE is not None:
        return _PINNED_REF_CACHE, (_PINNED_REF_SOURCE or "cache")

    ref_path, src_tag = _resolve_ref_h5ad_path()
    if ref_path is not None:
        # Use backed mode to avoid loading full reference matrix into RAM.
        ref = ad.read_h5ad(ref_path, backed="r")
        src = f"{src_tag}:{str(ref_path)}:backed=r"
    else:
        dest_dir = _zmap_home_dir() / "h5ads"
        dest_dir.mkdir(parents=True, exist_ok=True)
        ref = load_zmap_h5ad(
            kind=DEFAULT_REF_KIND_FALLBACK,
            dest_dir=dest_dir,
            use_cache=True,
            show_progress=False,
            backed="r",
        )
        src = (
            f"zmap.reference.load_zmap_h5ad(kind='{DEFAULT_REF_KIND_FALLBACK}', "
            f"dest_dir='{str(dest_dir)}', backed='r')"
        )

    if use_cache:
        _PINNED_REF_CACHE = ref
        _PINNED_REF_SOURCE = src
    return ref, src


def _get_layer(adata: ad.AnnData, layer_key: str):
    if layer_key == "X":
        return adata.X
    if layer_key not in adata.layers:
        raise KeyError(f"Missing query layer: {layer_key}")
    return adata.layers[layer_key]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (x / norms).astype(np.float32, copy=False)


def _hvg_mask_from_file(model_dir: Path, hvg_genes: np.ndarray) -> np.ndarray | None:
    mask_file = model_dir / "mask_hvg_genes.txt"
    if not mask_file.exists():
        return None
    genes = [g.strip() for g in mask_file.read_text().splitlines() if g.strip()]
    if not genes:
        return None
    gene_set = set(genes)
    mask = np.array([g in gene_set for g in hvg_genes], dtype=bool)
    return mask if np.any(mask) else None


class _MLP:
    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        out_dim: int,
        *,
        use_layernorm: bool,
        hidden_layernorm: bool,
        dropout: float,
    ) -> None:
        import torch

        ln_in = torch.nn.LayerNorm(in_dim) if use_layernorm else None
        dims = [in_dim] + list(hidden) + [out_dim]
        layers: list[torch.nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if hidden_layernorm:
                    layers.append(torch.nn.LayerNorm(dims[i + 1]))
                layers.append(torch.nn.GELU())
                if dropout and dropout > 0:
                    layers.append(torch.nn.Dropout(dropout))
        net = torch.nn.Sequential(*layers)
        self.module = torch.nn.Module()
        self.module.ln_in = ln_in
        self.module.net = net

        def _forward(x):
            if self.module.ln_in is not None:
                x = self.module.ln_in(x)
            return self.module.net(x)

        self.forward = _forward


def apply_encoder(
    adata_query: ad.AnnData,
    *,
    layer_key: str = "raw_nolog",
    batch_size: int = 4096,
    device: str = "cpu",
    latent_out_key: str | None = None,
) -> ad.AnnData:
    """
    Project query expression to encoder latent space using a pinned model directory.
    """
    import torch

    model_dir, model_source = _resolve_encoder_model_dir()
    cfg_path, ckpt_path = _ensure_model_files(model_dir)

    with cfg_path.open() as f:
        cfg = json.load(f)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    hvg_genes = ckpt.get("hvg_genes", None)
    if hvg_genes is None:
        raise ValueError("Checkpoint missing hvg_genes; cannot align query genes.")
    hvg_genes = np.array(hvg_genes, dtype=object)
    latent_dim = int(ckpt["latent_dim"])
    latent_base_key = str(cfg.get("latent_key", "X_pca")).strip() or "X_pca"
    if latent_out_key is None or str(latent_out_key).strip() == "":
        latent_out_key = f"{latent_base_key}_pred"
    latent_out_key = str(latent_out_key)

    hidden = cfg.get("hidden", [1024, 512])
    use_layernorm = bool(
        ckpt.get("input_layernorm", cfg.get("input_layernorm", cfg.get("use_layernorm", False)))
    )
    if "hidden_layernorm" in ckpt:
        hidden_layernorm = bool(ckpt["hidden_layernorm"])
    elif "hidden_layernorm" in cfg:
        hidden_layernorm = bool(cfg["hidden_layernorm"])
    elif "no_hidden_layernorm" in ckpt:
        hidden_layernorm = not bool(ckpt["no_hidden_layernorm"])
    elif "no_hidden_layernorm" in cfg:
        hidden_layernorm = not bool(cfg["no_hidden_layernorm"])
    else:
        sd_keys = list(ckpt.get("state_dict", {}).keys())
        if any(k.startswith("net.1.") for k in sd_keys):
            hidden_layernorm = True
        elif any(k.startswith("net.2.") for k in sd_keys):
            hidden_layernorm = False
        else:
            hidden_layernorm = True
    dropout = float(cfg.get("dropout", ckpt.get("dropout", 0.0)))

    def _build_mlp(hidden_ln: bool) -> _MLP:
        return _MLP(
            len(hvg_genes),
            hidden,
            latent_dim,
            use_layernorm=use_layernorm,
            hidden_layernorm=hidden_ln,
            dropout=dropout,
        )

    mlp = _build_mlp(hidden_layernorm)
    try:
        mlp.module.load_state_dict(ckpt["state_dict"])
    except RuntimeError as e_first:
        alt_hidden_layernorm = not bool(hidden_layernorm)
        mlp_alt = _build_mlp(alt_hidden_layernorm)
        try:
            mlp_alt.module.load_state_dict(ckpt["state_dict"])
            mlp = mlp_alt
            hidden_layernorm = alt_hidden_layernorm
            print(
                "[ZMAP] apply_encoder: recovered from architecture mismatch by "
                f"switching hidden_layernorm={hidden_layernorm}."
            )
        except RuntimeError:
            raise e_first
    run_device = device
    if str(run_device).startswith("cuda") and not torch.cuda.is_available():
        run_device = "cpu"
    mlp.module.to(run_device)
    mlp.module.eval()

    query_index = pd.Index(adata_query.var_names.astype(str))
    hvg_indexer = query_index.get_indexer(hvg_genes)
    present_mask = hvg_indexer >= 0
    present_idx = hvg_indexer[present_mask].astype(int)
    present_pos = np.where(present_mask)[0]
    mask_hvg = _hvg_mask_from_file(model_dir, hvg_genes)

    do_log1p = True
    do_normalize = bool(ckpt.get("input_normalize", False))
    target_sum = float(ckpt.get("input_target_sum", 1e6))
    do_scale = bool(ckpt.get("input_scale", False))
    scale_mean = ckpt.get("input_scale_mean", None)
    scale_std = ckpt.get("input_scale_std", None)
    if do_scale and (scale_mean is None or scale_std is None):
        raise ValueError("Encoder checkpoint expects input scaling but scale stats are missing.")
    if scale_mean is not None:
        scale_mean = np.asarray(scale_mean, dtype=np.float32)
    if scale_std is not None:
        scale_std = np.asarray(scale_std, dtype=np.float32)

    layer = _get_layer(adata_query, layer_key)
    n_obs = adata_query.n_obs
    latent = np.zeros((n_obs, latent_dim), dtype=np.float32)

    for start in range(0, n_obs, int(batch_size)):
        stop = min(n_obs, start + int(batch_size))
        batch_idx = np.arange(start, stop)
        X = np.zeros((len(batch_idx), len(hvg_genes)), dtype=np.float32)

        if present_idx.size > 0:
            Xp = layer[batch_idx][:, present_idx]
            if sp.issparse(Xp):
                Xp = Xp.toarray()
            Xp = np.asarray(Xp, dtype=np.float32)
            if do_normalize:
                full = layer[batch_idx]
                if sp.issparse(full):
                    lib = np.asarray(full.sum(axis=1)).ravel()
                else:
                    lib = np.asarray(np.sum(full, axis=1)).ravel()
                scale = np.zeros_like(lib, dtype=np.float32)
                nz = lib > 0
                scale[nz] = target_sum / lib[nz]
                Xp = Xp * scale[:, None]
            if do_log1p:
                Xp = np.log1p(Xp)
            X[:, present_pos] = Xp

        if do_scale and scale_mean is not None and scale_std is not None:
            std_safe = np.where(scale_std == 0, 1.0, scale_std)
            X = (X - scale_mean) / std_safe
        if mask_hvg is not None:
            X[:, mask_hvg] = 0.0

        with torch.no_grad():
            xt = torch.from_numpy(X).to(run_device)
            pred = mlp.forward(xt).detach().cpu().numpy().astype(np.float32, copy=False)
        latent[batch_idx] = pred

    adata_query.obsm[latent_out_key] = latent
    adata_query.uns.setdefault("zmap_encoder", {})
    adata_query.uns["zmap_encoder"].update(
        {
            "model_dir": str(model_dir),
            "model_source": model_source,
            "layer_key": layer_key,
            "device_requested": device,
            "device_used": run_device,
            "latent_out_key": latent_out_key,
            "n_hvg_model": int(len(hvg_genes)),
            "n_hvg_present_in_query": int(present_idx.size),
        }
    )
    return adata_query


def _knn_vote(
    ref_labels: np.ndarray,
    idx: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_query = idx.shape[0]
    pred = np.full(n_query, "unassigned", dtype=object)
    max_prob = np.zeros(n_query, dtype=np.float32)
    margin = np.zeros(n_query, dtype=np.float32)
    entropy = np.zeros(n_query, dtype=np.float32)
    mean_dist = np.full(n_query, np.nan, dtype=np.float32)
    for i in range(n_query):
        row_idx = idx[i]
        valid = row_idx >= 0
        if not np.any(valid):
            continue
        row_idx = row_idx[valid]
        row_dist = dist[i][valid]
        weights = 1.0 / (row_dist + 1e-8)
        total = float(weights.sum())
        if total <= 0:
            continue
        label_weights: dict[str, float] = {}
        for lbl, w in zip(ref_labels[row_idx], weights):
            label_weights[str(lbl)] = label_weights.get(str(lbl), 0.0) + float(w)
        probs = np.array(list(label_weights.values()), dtype=np.float64) / total
        pred[i] = max(label_weights.items(), key=lambda x: x[1])[0]
        max_prob[i] = float(np.max(probs))
        if probs.size >= 2:
            top2 = np.partition(probs, -2)[-2:]
            margin[i] = float(np.max(top2) - np.min(top2))
        else:
            margin[i] = float(max_prob[i])
        entropy[i] = float(-np.sum(probs * np.log(probs + 1e-12)))
        mean_dist[i] = float(np.mean(row_dist))
    return pred, max_prob, margin, entropy, mean_dist


def _apply_class_prior_distance_shift(
    idx: np.ndarray,
    dist: np.ndarray,
    *,
    ref_labels: np.ndarray,
    alpha: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply class-prior distance adjustment:
        D' = D + alpha * log(p(class))
    where p(class) is estimated from reference label frequencies.
    """
    if float(alpha) == 0.0:
        return np.asarray(dist, dtype=np.float32)
    labels = np.asarray(ref_labels, dtype=object).astype(str)
    if labels.size == 0:
        return np.asarray(dist, dtype=np.float32)
    vc = pd.Series(labels).value_counts(normalize=True, sort=False)
    logp = {str(k): float(np.log(max(float(v), float(eps)))) for k, v in vc.items()}

    d_out = np.asarray(dist, dtype=np.float32).copy()
    n_query, k = d_out.shape
    for i in range(n_query):
        row_idx = idx[i]
        valid = row_idx >= 0
        if not np.any(valid):
            continue
        row_ref = row_idx[valid]
        row_lbl = labels[row_ref]
        shifts = np.array([logp.get(str(lbl), np.log(float(eps))) for lbl in row_lbl], dtype=np.float32)
        d_out[i, valid] = d_out[i, valid] + float(alpha) * shifts
    return d_out


class _ReducerCompatSearchIndex:
    """
    Compatibility search index used by legacy UMAP reducers when serialized
    pynndescent/numba search functions cannot be restored in the current env.
    """

    def __init__(
        self,
        ref_data: np.ndarray,
        *,
        metric: str,
        n_neighbors_hint: int,
    ) -> None:
        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception as e:
            raise RuntimeError(f"Reducer compatibility search index requires sklearn: {e}") from e

        metric_use = str(metric).lower().strip()
        if metric_use in {"cosine", "correlation"}:
            self._angular_trees = True
        else:
            self._angular_trees = False
        if metric_use not in {"euclidean", "cosine", "correlation"}:
            metric_use = "euclidean"
            self._angular_trees = False

        self._nn = NearestNeighbors(
            n_neighbors=max(1, int(n_neighbors_hint)),
            metric=metric_use,
        )
        self._nn.fit(np.asarray(ref_data, dtype=np.float32))

    def query(
        self,
        x: np.ndarray,
        n_neighbors: int,
        epsilon: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        # epsilon is ignored (kept for API parity with pynndescent index).
        del epsilon
        dist, idx = self._nn.kneighbors(
            np.asarray(x, dtype=np.float32),
            n_neighbors=max(1, int(n_neighbors)),
            return_distance=True,
        )
        return (
            idx.astype(np.int32, copy=False),
            dist.astype(np.float32, copy=False),
        )


def _load_umap_reducer_with_compat(
    reducer_path: Path,
    *,
    mmap_mode: str | None = "r",
):
    """
    Load UMAP reducer joblib with legacy numpy random / pynndescent compat.

    Returns:
      reducer, load_mode
    """
    try:
        import joblib
    except Exception as e:
        raise RuntimeError(f"zmap_projection requires joblib: {e}") from e

    try:
        reducer = joblib.load(str(reducer_path), mmap_mode=mmap_mode)
        return reducer, "direct"
    except Exception as e_direct:
        try:
            import numpy.random as _np_random
            import numpy.random._pickle as _np_pickle  # type: ignore[attr-defined]
            import numpy.random._mt19937 as _np_mt19937  # type: ignore[attr-defined]
            import pynndescent.pynndescent_ as _pnd
        except Exception as e_dep:
            raise RuntimeError(
                "Failed to load UMAP reducer joblib and compat modules unavailable. "
                f"path={str(reducer_path)}; error={e_direct}; compat_import_error={e_dep}"
            ) from e_direct

        orig_bitgen_ctor = getattr(_np_pickle, "__bit_generator_ctor", None)
        orig_rs_ctor = getattr(_np_pickle, "__randomstate_ctor", None)
        orig_nnd_setstate = getattr(_pnd.NNDescent, "__setstate__", None)
        if not callable(orig_bitgen_ctor) or not callable(orig_rs_ctor) or not callable(orig_nnd_setstate):
            raise RuntimeError(
                "Failed to load UMAP reducer joblib: required compat hooks unavailable "
                f"in numpy/pynndescent. path={str(reducer_path)}; error={e_direct}"
            ) from e_direct

        class _CompatMT19937(_np_mt19937.MT19937):
            pass

        # Keep legacy internal state checks happy (expects class name MT19937).
        _CompatMT19937.__name__ = "MT19937"

        def _compat_mt_setstate(self, state):
            state_use = state
            if (
                isinstance(state_use, tuple)
                and len(state_use) >= 1
                and isinstance(state_use[0], dict)
            ):
                state_use = state_use[0]
            return _np_mt19937.MT19937.__setstate__(self, state_use)

        _CompatMT19937.__setstate__ = _compat_mt_setstate  # type: ignore[method-assign]

        def _compat_bitgen_ctor(name="MT19937"):
            if isinstance(name, _np_mt19937.MT19937):
                return name
            if not isinstance(name, str):
                name = getattr(name, "__name__", str(name))
            if str(name) == "MT19937":
                return _CompatMT19937()
            return orig_bitgen_ctor(name)

        def _compat_randomstate_ctor(
            bit_generator_name="MT19937",
            bit_generator_ctor=_compat_bitgen_ctor,
        ):
            return _np_random.RandomState(bit_generator_ctor(bit_generator_name))

        def _compat_nnd_setstate(self, d):
            self.__dict__ = d
            self._set_distance_func()
            self._search_forest = tuple(
                [_pnd.renumbaify_tree(tree) for tree in d["_search_forest"]]
            )
            
            # attach a compat search index below.

        setattr(_np_pickle, "__bit_generator_ctor", _compat_bitgen_ctor)
        setattr(_np_pickle, "__randomstate_ctor", _compat_randomstate_ctor)
        setattr(_pnd.NNDescent, "__setstate__", _compat_nnd_setstate)
        try:
            reducer = joblib.load(str(reducer_path), mmap_mode=mmap_mode)
        except Exception as e_compat:
            raise RuntimeError(
                "Failed to load UMAP reducer joblib with compat patch. "
                f"path={str(reducer_path)}; direct_error={e_direct}; compat_error={e_compat}"
            ) from e_compat
        finally:
            setattr(_np_pickle, "__bit_generator_ctor", orig_bitgen_ctor)
            setattr(_np_pickle, "__randomstate_ctor", orig_rs_ctor)
            setattr(_pnd.NNDescent, "__setstate__", orig_nnd_setstate)

        return reducer, "legacy_pickle_compat"


def _ensure_reducer_search_index_compat(
    reducer,
) -> str:
    """
    Ensure reducer._knn_search_index is query-able in current environment.
    """
    if not hasattr(reducer, "_raw_data"):
        return "not_applicable_no_raw_data"
    if str(getattr(reducer, "metric", "")).lower() == "precomputed":
        return "not_applicable_precomputed_metric"

    ref = np.asarray(getattr(reducer, "_raw_data"), dtype=np.float32)
    if ref.ndim != 2 or ref.shape[0] <= 0:
        return "not_applicable_bad_raw_data"

    k = int(getattr(reducer, "n_neighbors", 15) or 15)
    k = max(1, min(k, int(ref.shape[0])))

    idx_obj = getattr(reducer, "_knn_search_index", None)
    need_replace = (idx_obj is None) or (not hasattr(idx_obj, "query"))
    if not need_replace:
        try:
            probe = ref[:1]
            eps = 0.24 if bool(getattr(idx_obj, "_angular_trees", False)) else 0.12
            idx_obj.query(probe, k, epsilon=eps)
        except Exception:
            need_replace = True

    if need_replace:
        reducer._knn_search_index = _ReducerCompatSearchIndex(
            ref,
            metric=str(getattr(reducer, "metric", "euclidean")),
            n_neighbors_hint=k,
        )
        return "replaced_sklearn_knn_index"
    return "original_ok"


def zmap_projection(
    adata_query: ad.AnnData,
    *,
    query_latent_key: str = "X_pca_pred",
    out_umap_key: str = "X_umap_proj",
    umap_reducer_path: str | None = None,
    mmap_mode: str | None = "r",
) -> ad.AnnData:
    """
    Project query latent to ZMAP UMAP space using a prefit UMAP reducer joblib.

    Default reducer path:
      /c4/home/yingxins/ZMAP/output/phase3_trim_normalize/umap_reducer/umap_reducer.joblib
    """
    q_key = _resolve_query_latent_key_only(adata_query, query_latent_key=query_latent_key)
    reducer_path, reducer_source = _resolve_umap_reducer_path(umap_reducer_path)

    reducer, reducer_load_mode = _load_umap_reducer_with_compat(
        reducer_path,
        mmap_mode=mmap_mode,
    )
    reducer_index_mode = _ensure_reducer_search_index_compat(reducer)

    if not hasattr(reducer, "transform"):
        raise TypeError(
            f"Loaded object has no transform() method: type={type(reducer)} "
            f"path={str(reducer_path)}"
        )

    xq = np.asarray(adata_query.obsm[q_key], dtype=np.float32)
    n_feat_in = getattr(reducer, "n_features_in_", None)
    if n_feat_in is not None and int(xq.shape[1]) != int(n_feat_in):
        raise ValueError(
            "UMAP reducer input dimension mismatch: "
            f"query latent dim={xq.shape[1]} vs reducer n_features_in_={int(n_feat_in)}. "
            f"query_latent_key='{q_key}'"
        )

    try:
        umap_xy = reducer.transform(xq)
    except Exception as e:
        raise RuntimeError(f"UMAP reducer transform failed: {e}") from e

    umap_xy = np.asarray(umap_xy, dtype=np.float32)
    if umap_xy.ndim != 2 or umap_xy.shape[1] < 2:
        raise ValueError(f"Unexpected UMAP projection shape: {umap_xy.shape}")
    adata_query.obsm[str(out_umap_key)] = umap_xy
    adata_query.uns.setdefault("zmap_projection", {})
    adata_query.uns["zmap_projection"].update(
        {
            "umap_reducer_path": str(reducer_path),
            "umap_reducer_source": str(reducer_source),
            "query_latent_key": str(q_key),
            "out_umap_key": str(out_umap_key),
            "mmap_mode": (None if mmap_mode is None else str(mmap_mode)),
            "reducer_load_mode": str(reducer_load_mode),
            "reducer_index_mode": str(reducer_index_mode),
            "n_query": int(xq.shape[0]),
            "latent_dim": int(xq.shape[1]),
            "umap_dim": int(umap_xy.shape[1]),
        }
    )
    return adata_query


def _build_module_projection(
    module_csv: str,
    *,
    ref_var_names: np.ndarray,
    query_var_names: np.ndarray,
    min_genes_per_module: int,
) -> dict[str, object]:
    def _canon_gene(g: object) -> str:
        s = str(g).strip()
        return s.upper()

    def _first_index_map(
        names: np.ndarray,
    ) -> dict[str, int]:
        out: dict[str, int] = {}
        for i, g in enumerate(np.asarray(names, dtype=object).astype(str)):
            k = _canon_gene(g)
            if k and k not in out:
                out[k] = int(i)
        return out

    module_path = Path(module_csv)
    suffix = module_path.suffix.lower()

    ref_index = pd.Index(np.asarray(ref_var_names, dtype=object).astype(str))
    query_index = pd.Index(np.asarray(query_var_names, dtype=object).astype(str))
    shared_gene_index = ref_index.intersection(query_index)
    match_mode = "exact"
    ref_idx_map: dict[str, int] | None = None
    qry_idx_map: dict[str, int] | None = None
    shared_key_set: set[str] | None = None
    if shared_gene_index.size == 0:
        # Fallback 1: case-insensitive mapping.
        ref_idx_map = _first_index_map(ref_var_names)
        qry_idx_map = _first_index_map(query_var_names)
        shared_key_set = set(ref_idx_map).intersection(set(qry_idx_map))
        if shared_key_set:
            match_mode = "casefold"
        else:
            raise ValueError(
                "No shared genes between reference and query for module distance "
                "(tried exact and case-insensitive matching)."
            )

    if suffix in {".txt", ".tsv", ".list"}:
        genes = [g.strip() for g in module_path.read_text().splitlines() if g.strip()]
        if not genes:
            raise ValueError(f"Empty gene list file: {module_csv}")
        if match_mode == "exact":
            module_gene_index = pd.Index(pd.unique(np.asarray(genes, dtype=object)))
            used_gene_index = module_gene_index.intersection(shared_gene_index)
            if used_gene_index.size == 0:
                raise ValueError("No listed genes overlap shared ref/query genes.")
            n_genes = int(used_gene_index.size)
            genes_used = used_gene_index.astype(str).to_numpy()
            ref_gene_idx = ref_index.get_indexer(used_gene_index).astype(np.int64)
            query_gene_idx = query_index.get_indexer(used_gene_index).astype(np.int64)
        else:
            assert ref_idx_map is not None and qry_idx_map is not None and shared_key_set is not None
            canon = [
                _canon_gene(g)
                for g in genes
            ]
            canon_unique = list(dict.fromkeys([k for k in canon if k]))
            used_keys = [k for k in canon_unique if k in shared_key_set]
            if not used_keys:
                raise ValueError("No listed genes overlap shared ref/query genes (after normalization).")
            n_genes = len(used_keys)
            genes_used = np.asarray(used_keys, dtype=object)
            ref_gene_idx = np.asarray([ref_idx_map[k] for k in used_keys], dtype=np.int64)
            query_gene_idx = np.asarray([qry_idx_map[k] for k in used_keys], dtype=np.int64)

        if n_genes < int(min_genes_per_module):
            raise ValueError(
                f"Only {n_genes} genes overlap shared ref/query genes; "
                f"requires >= {int(min_genes_per_module)}."
            )
        # TXT gene-list mode: each selected gene is one feature axis.
        W = np.eye(n_genes, dtype=np.float32)
        module_names = [f"gene|{g}" for g in genes_used.astype(str)]
        module_levels = ["gene"] * n_genes
        module_gene_counts = [1] * n_genes
    else:
        df = pd.read_csv(module_csv)
        required = {"gene", "celltype"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"module CSV missing columns: {missing}")
        if "module_level" not in df.columns:
            df["module_level"] = "module"
        if "module_weight" not in df.columns:
            sr = (
                df["support_ratio"].astype(float).clip(lower=0.0)
                if "support_ratio" in df.columns
                else 1.0
            )
            enrich = (
                np.log1p(df["enrich_mean"].astype(float).clip(lower=0.0))
                if "enrich_mean" in df.columns
                else 1.0
            )
            lfc = (
                df["global_log2fc"].astype(float).clip(lower=0.0, upper=8.0)
                if "global_log2fc" in df.columns
                else 1.0
            )
            df["module_weight"] = (sr * enrich * lfc).astype(float)

        df["gene"] = df["gene"].astype(str)
        df["celltype"] = df["celltype"].astype(str)
        df["module_level"] = df["module_level"].astype(str)
        df["module_weight"] = df["module_weight"].astype(float).clip(lower=0.0)
        if match_mode != "exact":
            df["gene_key"] = [
                _canon_gene(g)
                for g in df["gene"].to_numpy(dtype=object)
            ]
            df = df[df["gene_key"] != ""]

        if match_mode == "exact":
            x = (
                df.groupby(["module_level", "celltype", "gene"], as_index=False)["module_weight"]
                .sum()
                .sort_values(
                    ["module_level", "celltype", "module_weight"],
                    ascending=[True, True, False],
                )
            )
            if x.shape[0] == 0:
                raise ValueError(f"No rows available after aggregation from module CSV: {module_csv}")
            module_gene_index = pd.Index(pd.unique(x["gene"]))
            used_gene_index = module_gene_index.intersection(shared_gene_index)
            if used_gene_index.size == 0:
                raise ValueError("No module genes overlap shared ref/query genes.")
            genes_used = used_gene_index.astype(str).to_numpy()
            gene_to_col = {g: i for i, g in enumerate(genes_used)}
            n_genes = len(gene_to_col)
            ref_gene_idx = ref_index.get_indexer(used_gene_index).astype(np.int64)
            query_gene_idx = query_index.get_indexer(used_gene_index).astype(np.int64)
        else:
            assert ref_idx_map is not None and qry_idx_map is not None and shared_key_set is not None
            x = (
                df.groupby(["module_level", "celltype", "gene_key"], as_index=False)["module_weight"]
                .sum()
                .sort_values(
                    ["module_level", "celltype", "module_weight"],
                    ascending=[True, True, False],
                )
            )
            if x.shape[0] == 0:
                raise ValueError(f"No rows available after aggregation from module CSV: {module_csv}")
            module_gene_keys = list(pd.unique(x["gene_key"]))
            used_keys = [k for k in module_gene_keys if k in shared_key_set]
            if not used_keys:
                raise ValueError("No module genes overlap shared ref/query genes (after normalization).")
            genes_used = np.asarray(used_keys, dtype=object)
            gene_to_col = {g: i for i, g in enumerate(used_keys)}
            n_genes = len(gene_to_col)
            ref_gene_idx = np.asarray([ref_idx_map[k] for k in used_keys], dtype=np.int64)
            query_gene_idx = np.asarray([qry_idx_map[k] for k in used_keys], dtype=np.int64)

        module_names = []
        module_levels = []
        module_gene_counts = []
        cols: list[np.ndarray] = []
        for (lvl, ct), g in x.groupby(["module_level", "celltype"], sort=False):
            if match_mode == "exact":
                gg = g[g["gene"].isin(pd.Index(genes_used))].copy()
                idx = np.array([gene_to_col[str(v)] for v in gg["gene"].astype(str)], dtype=np.int64)
            else:
                gg = g[g["gene_key"].isin(pd.Index(genes_used))].copy()
                idx = np.array([gene_to_col[str(v)] for v in gg["gene_key"].astype(str)], dtype=np.int64)
            if gg.empty:
                continue
            w = gg["module_weight"].to_numpy(dtype=np.float32)
            denom = float(np.sum(w))
            if denom <= 0:
                continue
            w = w / denom
            if idx.size < int(min_genes_per_module):
                continue
            col = np.zeros(n_genes, dtype=np.float32)
            col[idx] = w
            cols.append(col)
            module_names.append(f"{str(lvl)}|{str(ct)}")
            module_levels.append(str(lvl))
            module_gene_counts.append(int(idx.size))

        if not cols:
            raise ValueError(
                "No valid modules after shared-gene intersection and min-gene filter. "
                "Try a smaller module_min_genes or another module CSV."
            )

        W = np.stack(cols, axis=1).astype(np.float32, copy=False)  # [n_genes, n_modules]

    if np.any(ref_gene_idx < 0) or np.any(query_gene_idx < 0):
        raise RuntimeError("Internal error: module gene indexer contains missing entries.")

    return {
        "W": W,
        "module_names": module_names,
        "module_levels": module_levels,
        "module_gene_counts": module_gene_counts,
        "genes_used": np.asarray(genes_used, dtype=object).astype(str),
        "gene_match_mode": match_mode,
        "ref_gene_idx": ref_gene_idx,
        "query_gene_idx": query_gene_idx,
    }


def _compute_module_scores(
    adata: ad.AnnData,
    *,
    layer_key: str,
    gene_idx: np.ndarray,
    W: np.ndarray,
    row_idx: np.ndarray | None = None,
    batch_size: int = 8192,
) -> np.ndarray:
    layer = _get_layer(adata, layer_key)
    rows = (
        np.arange(adata.n_obs, dtype=np.int64)
        if row_idx is None
        else np.asarray(row_idx, dtype=np.int64)
    )
    n_rows = rows.size
    n_modules = int(W.shape[1])
    scores = np.zeros((n_rows, n_modules), dtype=np.float32)
    if n_rows == 0:
        return scores

    for start in range(0, n_rows, int(batch_size)):
        stop = min(n_rows, start + int(batch_size))
        r = rows[start:stop]
        X = layer[r][:, gene_idx]
        if sp.issparse(X):
            out = X @ W
            if sp.issparse(out):
                out = out.toarray()
        else:
            out = np.asarray(X, dtype=np.float32) @ W
        scores[start:stop] = np.asarray(out, dtype=np.float32)
    return scores


def _rerank_with_module_distance(
    idx: np.ndarray,
    dist: np.ndarray,
    *,
    ref_module_scores: np.ndarray,
    query_module_scores: np.ndarray,
    k: int,
    module_lambda: float,
    module_metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_query, n_cand = idx.shape
    if query_module_scores.shape[0] != n_query:
        raise ValueError(
            "query_module_scores shape mismatch: "
            f"{query_module_scores.shape[0]} vs n_query={n_query}"
        )
    if ref_module_scores.shape[1] != query_module_scores.shape[1]:
        raise ValueError(
            "Module score dimension mismatch: "
            f"ref={ref_module_scores.shape[1]} query={query_module_scores.shape[1]}"
        )

    metric = str(module_metric).lower()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError(f"Unsupported module_metric: {module_metric}")

    ref_feat = np.asarray(ref_module_scores, dtype=np.float32)
    qry_feat = np.asarray(query_module_scores, dtype=np.float32)
    if metric == "cosine":
        ref_feat = _l2_normalize(ref_feat)
        qry_feat = _l2_normalize(qry_feat)

    idx_out = np.full((n_query, int(k)), -1, dtype=np.int64)
    dist_out = np.full((n_query, int(k)), np.nan, dtype=np.float32)
    lam = float(module_lambda)

    for i in range(n_query):
        row_idx = idx[i]
        row_dist = dist[i]
        valid = (row_idx >= 0) & np.isfinite(row_dist)
        if not np.any(valid):
            continue
        cand_idx = row_idx[valid].astype(np.int64, copy=False)
        base_dist = row_dist[valid].astype(np.float32, copy=False)
        qv = qry_feat[i]

        if metric == "euclidean":
            rv = ref_feat[cand_idx]
            d_mod = np.sqrt(np.sum((rv - qv[None, :]) ** 2, axis=1)).astype(np.float32, copy=False)
        else:
            d_mod = np.maximum(
                1.0 - np.sum(ref_feat[cand_idx] * qv[None, :], axis=1),
                0.0,
            ).astype(np.float32, copy=False)

        d_total = base_dist + lam * d_mod
        order = np.argsort(d_total)[: int(k)]
        kk = int(order.size)
        idx_out[i, :kk] = cand_idx[order]
        dist_out[i, :kk] = d_total[order]

    return idx_out, dist_out


def build_module_faiss(
    adata_ref: ad.AnnData,
    out_dir: str,
    *,
    ref_latent_key: str = DEFAULT_REF_LATENT_KEY,
    label_col: str = DEFAULT_LABEL_COL,
    tissue_col: str = DEFAULT_TISSUE_COL,
    include_unassigned: bool = False,
    module_csv: str | None = None,
    module_layer_key: str = "raw_nolog",
    module_gene_key: str = "auto",
    module_metric: str = "cosine",
    module_lambda: float = 0.3,
    emb_l2norm: bool = True,
    module_l2norm: bool = True,
    index_type: str = "ivf_flat",
    nlist: int = 4096,
    nprobe: int = 16,
    overwrite: bool = False,
) -> dict[str, object]:
    """
    Build and persist a FAISS index on concatenated [embedding, sqrt(lambda)*module] features.

    This is an offline/precompute helper for reference data only.
    """
    if ref_latent_key not in adata_ref.obsm:
        raise KeyError(f"Missing ref latent key in adata_ref.obsm: {ref_latent_key}")
    if label_col not in adata_ref.obs:
        raise KeyError(f"Missing label column in adata_ref.obs: {label_col}")
    if float(module_lambda) < 0:
        raise ValueError("module_lambda must be >= 0.")
    if int(nlist) <= 0:
        raise ValueError("nlist must be positive.")
    if int(nprobe) <= 0:
        raise ValueError("nprobe must be positive.")

    idx_type = str(index_type).lower().strip()
    if idx_type not in {"flat", "ivf_flat"}:
        raise ValueError("index_type must be one of {'flat', 'ivf_flat'}.")

    mod_metric = str(module_metric).lower().strip()
    if mod_metric not in {"euclidean", "cosine"}:
        raise ValueError("module_metric must be one of {'euclidean', 'cosine'}.")

    if str(module_gene_key).strip() == "":
        raise ValueError("module_gene_key must be a non-empty string.")

    module_path, module_source = _resolve_module_markers_csv_path(module_csv)
    if module_path is None:
        raise FileNotFoundError(
            "No module marker file found. Set `module_csv`, or env `ZMAP_MODULE_CSV`, "
            "or place the default file under ZMAP_HOME/modules."
        )
    module_csv_norm = str(module_path.resolve())

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "faiss.index"
    meta_path = out / "meta.json"
    proj_path = out / "module_projection.npz"
    labels_path = out / "ref_labels.npy"
    tissue_path = out / "ref_tissue.npy"
    ref_rows_path = out / "ref_rows.npy"
    for p in [index_path, meta_path, proj_path, labels_path, tissue_path, ref_rows_path]:
        if p.exists() and (not bool(overwrite)):
            raise FileExistsError(
                f"Output already exists: {p}. Set overwrite=True to replace."
            )

    ref_labels_all = adata_ref.obs[label_col].astype(str).to_numpy()
    if include_unassigned:
        keep = np.ones(ref_labels_all.shape[0], dtype=bool)
    else:
        keep = ref_labels_all != "unassigned"
    ref_rows = np.flatnonzero(keep)
    if ref_rows.size == 0:
        raise ValueError("No reference rows left after include_unassigned filtering.")

    emb_all = np.asarray(adata_ref.obsm[ref_latent_key], dtype=np.float32)
    emb_ref = emb_all[keep]
    if bool(emb_l2norm):
        emb_ref = _l2_normalize(emb_ref)

    # Resolve which ref gene namespace to use when mapping module genes.
    gene_key_req = str(module_gene_key).strip()
    if gene_key_req == "":
        gene_key_req = "auto"
    candidates: list[tuple[str, np.ndarray]] = [("var_names", adata_ref.var_names.to_numpy())]
    if gene_key_req == "auto":
        for col in [
            "gene",
            "gene_name",
            "gene_symbol",
            "symbol",
            "feature_name",
            "gene_id",
        ]:
            if col in adata_ref.var.columns:
                candidates.append((f"var[{col}]", adata_ref.var[col].astype(str).to_numpy()))
    elif gene_key_req != "var_names":
        if gene_key_req not in adata_ref.var.columns:
            raise KeyError(f"module_gene_key='{gene_key_req}' not found in adata_ref.var.")
        candidates = [(f"var[{gene_key_req}]", adata_ref.var[gene_key_req].astype(str).to_numpy())]

    proj = None
    proj_source = None
    last_err: Exception | None = None
    for src_name, gene_vals in candidates:
        try:
            # For ref-only bundle, use the same gene namespace on both sides.
            proj = _build_module_projection(
                module_csv_norm,
                ref_var_names=gene_vals,
                query_var_names=gene_vals,
                min_genes_per_module=1,
            )
            proj_source = src_name
            break
        except Exception as e:
            last_err = e
            continue
    if proj is None:
        if last_err is not None:
            raise last_err
        raise RuntimeError("Failed to build module projection.")

    module_ref = _compute_module_scores(
        adata_ref,
        layer_key=str(module_layer_key),
        gene_idx=np.asarray(proj["ref_gene_idx"], dtype=np.int64),  # type: ignore[index]
        W=np.asarray(proj["W"], dtype=np.float32),  # type: ignore[index]
        row_idx=ref_rows,
    )
    if bool(module_l2norm):
        module_ref = _l2_normalize(module_ref)

    lam_sqrt = float(np.sqrt(float(module_lambda)))
    module_scaled = module_ref * lam_sqrt
    x_total = np.hstack([emb_ref, module_scaled]).astype(np.float32, copy=False)
    if x_total.shape[0] == 0 or x_total.shape[1] == 0:
        raise ValueError(f"Invalid concatenated feature shape: {x_total.shape}")

    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(f"FAISS import failed: {e}") from e

    index_used = "flat"
    nlist_eff = 1
    nprobe_eff = 1
    if idx_type == "flat":
        index = faiss.IndexFlatL2(x_total.shape[1])
        index.add(x_total)
    else:
        try:
            n_ref = int(x_total.shape[0])
            nlist_eff = max(1, min(int(nlist), n_ref, max(1, n_ref // 40)))
            nprobe_eff = max(1, min(int(nprobe), nlist_eff))
            quantizer = faiss.IndexFlatL2(x_total.shape[1])
            index = faiss.IndexIVFFlat(
                quantizer,
                x_total.shape[1],
                int(nlist_eff),
                faiss.METRIC_L2,
            )
            train_n = min(
                n_ref,
                max(
                    int(nlist_eff) * 40,
                    min(n_ref, 200000),
                ),
            )
            if train_n < n_ref:
                rng = np.random.default_rng(0)
                train_idx = rng.choice(n_ref, size=train_n, replace=False)
                train_x = x_total[train_idx]
            else:
                train_x = x_total
            index.train(np.ascontiguousarray(train_x, dtype=np.float32))
            index.add(x_total)
            index.nprobe = int(nprobe_eff)
            index_used = "ivf_flat"
        except Exception as e:
            index = faiss.IndexFlatL2(x_total.shape[1])
            index.add(x_total)
            index_used = "flat"
            nlist_eff = 1
            nprobe_eff = 1
            print(f"[ZMAP] build_module_faiss: IVF build failed ({e}), fallback to flat.")

    faiss.write_index(index, str(index_path))

    np.save(str(labels_path), np.asarray(ref_labels_all[keep], dtype=object), allow_pickle=True)
    if tissue_col in adata_ref.obs:
        ref_tissue = adata_ref.obs[tissue_col].astype(str).to_numpy()[keep]
    else:
        ref_tissue = np.full(ref_rows.size, "", dtype=object)
    np.save(str(tissue_path), np.asarray(ref_tissue, dtype=object), allow_pickle=True)
    np.save(str(ref_rows_path), ref_rows.astype(np.int64))
    np.savez_compressed(
        str(proj_path),
        W=np.asarray(proj["W"], dtype=np.float32),  # type: ignore[index]
        ref_gene_idx=np.asarray(proj["ref_gene_idx"], dtype=np.int64),  # type: ignore[index]
        genes_used=np.asarray(proj["genes_used"], dtype=object),  # type: ignore[index]
        module_names=np.asarray(proj["module_names"], dtype=object),  # type: ignore[index]
        module_levels=np.asarray(proj["module_levels"], dtype=object),  # type: ignore[index]
        module_gene_counts=np.asarray(proj["module_gene_counts"], dtype=np.int64),  # type: ignore[index]
    )

    meta = {
        "index_path": str(index_path),
        "projection_path": str(proj_path),
        "labels_path": str(labels_path),
        "tissue_path": str(tissue_path),
        "ref_rows_path": str(ref_rows_path),
        "n_ref_used": int(ref_rows.size),
        "d_embedding": int(emb_ref.shape[1]),
        "d_module": int(module_ref.shape[1]),
        "d_total": int(x_total.shape[1]),
        "ref_latent_key": str(ref_latent_key),
        "label_col": str(label_col),
        "tissue_col": str(tissue_col),
        "include_unassigned": bool(include_unassigned),
        "module_csv": module_csv_norm,
        "module_csv_source": str(module_source),
        "module_gene_key_requested": str(module_gene_key),
        "module_gene_source": str(proj_source),
        "gene_match_mode": str(proj.get("gene_match_mode", "exact")),
        "module_layer_key": str(module_layer_key),
        "module_metric": str(mod_metric),
        "module_lambda": float(module_lambda),
        "emb_l2norm": bool(emb_l2norm),
        "module_l2norm": bool(module_l2norm),
        "index_type_requested": str(idx_type),
        "index_type_used": str(index_used),
        "nlist_requested": int(nlist),
        "nlist_used": int(nlist_eff),
        "nprobe_requested": int(nprobe),
        "nprobe_used": int(nprobe_eff),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(
        "[ZMAP] build_module_faiss done: "
        f"n_ref={meta['n_ref_used']} d_total={meta['d_total']} "
        f"index={meta['index_type_used']} out={str(out)}"
    )
    return meta


def build_module_concat_reference(
    adata_ref: ad.AnnData,
    *,
    ref_latent_key: str = DEFAULT_REF_LATENT_KEY,
    out_latent_key: str = "X_pca_module",
    module_csv: str | None = None,
    module_layer_key: str = "raw_nolog",
    module_gene_key: str = "auto",
    module_lambda: float = 0.3,
    emb_l2norm: bool = True,
    module_l2norm: bool = True,
    projection_key: str = "zmap_module_projection",
) -> ad.AnnData:
    """
    Build concatenated [embedding, sqrt(lambda)*module] latent for reference AnnData.

    Saves:
      - adata_ref.obsm[out_latent_key]
      - adata_ref.uns[projection_key]  (module projection metadata + weights)
    """
    if ref_latent_key not in adata_ref.obsm:
        raise KeyError(f"Missing ref latent key in adata_ref.obsm: {ref_latent_key}")
    if float(module_lambda) < 0:
        raise ValueError("module_lambda must be >= 0.")
    if str(module_gene_key).strip() == "":
        raise ValueError("module_gene_key must be a non-empty string.")

    module_path, module_source = _resolve_module_markers_csv_path(module_csv)
    if module_path is None:
        raise FileNotFoundError(
            "No module marker file found. Set `module_csv`, or env `ZMAP_MODULE_CSV`, "
            "or place the default file under ZMAP_HOME/modules."
        )
    module_csv_norm = str(module_path.resolve())

    gene_key_req = str(module_gene_key).strip()
    if gene_key_req == "":
        gene_key_req = "auto"
    candidates: list[tuple[str, np.ndarray]] = [("var_names", adata_ref.var_names.to_numpy())]
    if gene_key_req == "auto":
        for col in [
            "gene",
            "gene_name",
            "gene_symbol",
            "symbol",
            "feature_name",
            "gene_id",
        ]:
            if col in adata_ref.var.columns:
                candidates.append((f"var[{col}]", adata_ref.var[col].astype(str).to_numpy()))
    elif gene_key_req != "var_names":
        if gene_key_req not in adata_ref.var.columns:
            raise KeyError(f"module_gene_key='{gene_key_req}' not found in adata_ref.var.")
        candidates = [(f"var[{gene_key_req}]", adata_ref.var[gene_key_req].astype(str).to_numpy())]

    proj = None
    proj_source = None
    last_err: Exception | None = None
    for src_name, gene_vals in candidates:
        try:
            proj = _build_module_projection(
                module_csv_norm,
                ref_var_names=gene_vals,
                query_var_names=gene_vals,
                min_genes_per_module=1,
            )
            proj_source = src_name
            break
        except Exception as e:
            last_err = e
            continue
    if proj is None:
        if last_err is not None:
            raise last_err
        raise RuntimeError("Failed to build module projection.")

    module_ref = _compute_module_scores(
        adata_ref,
        layer_key=str(module_layer_key),
        gene_idx=np.asarray(proj["ref_gene_idx"], dtype=np.int64),  # type: ignore[index]
        W=np.asarray(proj["W"], dtype=np.float32),  # type: ignore[index]
        row_idx=None,
    )
    if bool(module_l2norm):
        module_ref = _l2_normalize(module_ref)

    emb_ref = np.asarray(adata_ref.obsm[ref_latent_key], dtype=np.float32)
    if bool(emb_l2norm):
        emb_ref = _l2_normalize(emb_ref)

    x_total = np.hstack(
        [
            emb_ref,
            module_ref * float(np.sqrt(float(module_lambda))),
        ]
    ).astype(np.float32, copy=False)
    adata_ref.obsm[out_latent_key] = x_total

    adata_ref.uns[projection_key] = {
        "module_csv": module_csv_norm,
        "module_csv_source": str(module_source),
        "module_gene_key_requested": str(module_gene_key),
        "module_gene_source": str(proj_source),
        "gene_match_mode": str(proj.get("gene_match_mode", "exact")),
        "module_layer_key": str(module_layer_key),
        "module_lambda": float(module_lambda),
        "emb_l2norm": bool(emb_l2norm),
        "module_l2norm": bool(module_l2norm),
        "ref_latent_key": str(ref_latent_key),
        "out_latent_key": str(out_latent_key),
        "W": np.asarray(proj["W"], dtype=np.float32),  # type: ignore[index]
        "genes_used": np.asarray(proj["genes_used"], dtype=object),  # type: ignore[index]
        "module_names": np.asarray(proj["module_names"], dtype=object),  # type: ignore[index]
        "module_levels": np.asarray(proj["module_levels"], dtype=object),  # type: ignore[index]
        "module_gene_counts": np.asarray(proj["module_gene_counts"], dtype=np.int64),  # type: ignore[index]
    }
    return adata_ref


def apply_module_concat_query(
    adata_query: ad.AnnData,
    adata_ref: ad.AnnData,
    *,
    query_latent_key: str = "X_pca_pred",
    out_latent_key: str = "X_pca_pred_module",
    projection_key: str = "zmap_module_projection",
    module_layer_key: str | None = None,
    module_lambda: float | None = None,
    emb_l2norm: bool | None = None,
    module_l2norm: bool | None = None,
    module_gene_key: str = "auto",
) -> ad.AnnData:
    """
    Build concatenated [embedding, sqrt(lambda)*module] latent for query AnnData
    using projection metadata stored in `adata_ref.uns[projection_key]`.
    """
    if query_latent_key not in adata_query.obsm:
        raise KeyError(f"Missing query latent key in adata_query.obsm: {query_latent_key}")
    if projection_key not in adata_ref.uns:
        raise KeyError(
            f"Missing projection_key='{projection_key}' in adata_ref.uns. "
            "Run build_module_concat_reference(...) first."
        )
    proj = adata_ref.uns[projection_key]
    if not isinstance(proj, dict):
        raise TypeError(f"adata_ref.uns['{projection_key}'] must be a dict.")
    for k in ["W", "genes_used"]:
        if k not in proj:
            raise KeyError(f"Missing '{k}' in adata_ref.uns['{projection_key}'].")

    w = np.asarray(proj["W"], dtype=np.float32)
    genes_used = np.asarray(proj["genes_used"], dtype=object).astype(str)
    gene_match_mode = str(proj.get("gene_match_mode", "exact"))

    if module_layer_key is None:
        module_layer_key = str(proj.get("module_layer_key", "raw_nolog"))
    if module_lambda is None:
        module_lambda = float(proj.get("module_lambda", 0.0))
    if emb_l2norm is None:
        emb_l2norm = bool(proj.get("emb_l2norm", True))
    if module_l2norm is None:
        module_l2norm = bool(proj.get("module_l2norm", True))
    if float(module_lambda) < 0:
        raise ValueError("module_lambda must be >= 0.")

    gene_source_ref = str(proj.get("module_gene_source", "var_names"))
    gene_key_req = str(module_gene_key).strip()
    if gene_key_req == "" or gene_key_req == "auto":
        gene_source = gene_source_ref
    elif gene_key_req == "var_names":
        gene_source = "var_names"
    else:
        gene_source = f"var[{gene_key_req}]"

    if gene_source == "var_names":
        q_genes = np.asarray(adata_query.var_names.astype(str), dtype=object)
    elif gene_source.startswith("var[") and gene_source.endswith("]"):
        col = gene_source[4:-1]
        if col not in adata_query.var.columns:
            raise KeyError(
                f"Required query var column '{col}' not found for module projection."
            )
        q_genes = np.asarray(adata_query.var[col].astype(str), dtype=object)
    else:
        raise ValueError(f"Unsupported module gene source: {gene_source}")

    if gene_match_mode == "exact":
        idx = pd.Index(q_genes).get_indexer(genes_used).astype(np.int64)
    elif gene_match_mode == "casefold":
        q_map: dict[str, int] = {}
        for i, g in enumerate(q_genes.astype(str)):
            k = g.strip().upper()
            if k and k not in q_map:
                q_map[k] = int(i)
        idx = np.asarray([q_map.get(g.strip().upper(), -1) for g in genes_used], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported gene_match_mode in projection: {gene_match_mode}")

    present = idx >= 0
    n_present = int(np.sum(present))
    if n_present <= 0:
        raise ValueError(
            "No module genes from reference projection are present in query features."
        )
    if n_present < len(genes_used):
        print(
            f"[ZMAP] apply_module_concat_query: using {n_present}/{len(genes_used)} module genes present in query."
        )

    w_eff = w[present]
    idx_eff = idx[present]
    module_q = _compute_module_scores(
        adata_query,
        layer_key=str(module_layer_key),
        gene_idx=np.asarray(idx_eff, dtype=np.int64),
        W=np.asarray(w_eff, dtype=np.float32),
        row_idx=None,
    )
    if bool(module_l2norm):
        module_q = _l2_normalize(module_q)

    emb_q = np.asarray(adata_query.obsm[query_latent_key], dtype=np.float32)
    if bool(emb_l2norm):
        emb_q = _l2_normalize(emb_q)

    x_total = np.hstack(
        [
            emb_q,
            module_q * float(np.sqrt(float(module_lambda))),
        ]
    ).astype(np.float32, copy=False)
    adata_query.obsm[out_latent_key] = x_total
    adata_query.uns.setdefault("zmap_module_concat_query", {})
    adata_query.uns["zmap_module_concat_query"].update(
        {
            "projection_key": str(projection_key),
            "out_latent_key": str(out_latent_key),
            "query_latent_key": str(query_latent_key),
            "module_layer_key": str(module_layer_key),
            "module_lambda": float(module_lambda),
            "emb_l2norm": bool(emb_l2norm),
            "module_l2norm": bool(module_l2norm),
            "gene_source_used": str(gene_source),
            "gene_match_mode": str(gene_match_mode),
            "n_module_genes_total": int(len(genes_used)),
            "n_module_genes_present_in_query": int(n_present),
            "d_total": int(x_total.shape[1]),
        }
    )
    return adata_query


def _knn_global(
    ref_latent: np.ndarray,
    query_latent: np.ndarray,
    *,
    k: int,
    metric: str,
    knn_backend: str,
    knn_device: str,
    knn_nprobe: int | None,
    faiss_cache_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    idx, dist, _ = knn_search(
        ref_latent,
        query_latent,
        n_neighbors=int(k),
        metric=metric,
        backend=knn_backend,
        device=knn_device,
        nprobe=knn_nprobe,
        cache_key=faiss_cache_key,
    )
    return idx, dist.astype(np.float32, copy=False)


def _knn_hard_tissue(
    ref_latent: np.ndarray,
    ref_tissue: np.ndarray,
    query_latent: np.ndarray,
    query_tissue: np.ndarray,
    *,
    k: int,
    metric: str,
    knn_backend: str,
    knn_device: str,
    knn_nprobe: int | None,
    faiss_cache_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_query = query_latent.shape[0]
    idx_out = np.full((n_query, k), -1, dtype=np.int64)
    dist_out = np.full((n_query, k), np.nan, dtype=np.float32)

    global_idx = None
    global_dist = None
    for tissue in np.unique(query_tissue):
        q_rows = np.flatnonzero(query_tissue == tissue)
        if q_rows.size == 0:
            continue
        r_rows = np.flatnonzero(ref_tissue == tissue)
        if r_rows.size >= k:
            idx_local, dist_local = _knn_global(
                ref_latent[r_rows],
                query_latent[q_rows],
                k=k,
                metric=metric,
                knn_backend=knn_backend,
                knn_device=knn_device,
                knn_nprobe=knn_nprobe,
                faiss_cache_key=(
                    None
                    if faiss_cache_prefix is None
                    else f"{str(faiss_cache_prefix)}|tissue={str(tissue)}"
                ),
            )
            mapped = np.full_like(idx_local, -1)
            ok = idx_local >= 0
            if np.any(ok):
                mapped[ok] = r_rows[idx_local[ok]]
            idx_out[q_rows] = mapped
            dist_out[q_rows] = dist_local
        else:
            if global_idx is None or global_dist is None:
                global_idx, global_dist = _knn_global(
                    ref_latent,
                    query_latent,
                    k=k,
                    metric=metric,
                    knn_backend=knn_backend,
                    knn_device=knn_device,
                    knn_nprobe=knn_nprobe,
                    faiss_cache_key=(
                        None
                        if faiss_cache_prefix is None
                        else f"{str(faiss_cache_prefix)}|global"
                    ),
                )
            idx_out[q_rows] = global_idx[q_rows]
            dist_out[q_rows] = global_dist[q_rows]
    return idx_out, dist_out


def _knn_soft_tissue(
    ref_latent: np.ndarray,
    ref_tissue: np.ndarray,
    query_latent: np.ndarray,
    query_tissue: np.ndarray,
    *,
    k: int,
    metric: str,
    penalty_lambda: float,
    knn_backend: str,
    knn_device: str,
    knn_nprobe: int | None,
    faiss_cache_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_query = query_latent.shape[0]
    idx_out = np.full((n_query, k), -1, dtype=np.int64)
    dist_out = np.full((n_query, k), np.nan, dtype=np.float32)

    global_k = min(max(k * 3, k + 5), ref_latent.shape[0])
    g_idx, g_dist = _knn_global(
        ref_latent,
        query_latent,
        k=global_k,
        metric=metric,
        knn_backend=knn_backend,
        knn_device=knn_device,
        knn_nprobe=knn_nprobe,
        faiss_cache_key=(
            None
            if faiss_cache_prefix is None
            else f"{str(faiss_cache_prefix)}|global"
        ),
    )

    tissue_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tissue in np.unique(query_tissue):
        r_rows = np.flatnonzero(ref_tissue == tissue)
        if r_rows.size == 0:
            continue
        lk = min(k, int(r_rows.size))
        li, ld = _knn_global(
            ref_latent[r_rows],
            query_latent[query_tissue == tissue],
            k=lk,
            metric=metric,
            knn_backend=knn_backend,
            knn_device=knn_device,
            knn_nprobe=knn_nprobe,
            faiss_cache_key=(
                None
                if faiss_cache_prefix is None
                else f"{str(faiss_cache_prefix)}|tissue={str(tissue)}"
            ),
        )
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
                penalty = 0.0 if ref_tissue[ridx] == tissue else float(penalty_lambda)
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
                if ridx in best:
                    best[ridx] = min(best[ridx], d)
                else:
                    best[ridx] = d
            cidx = np.fromiter(best.keys(), dtype=np.int64)
            cdist = np.fromiter(best.values(), dtype=np.float32)
            order = np.argsort(cdist)[:k]
            k_use = order.size
            idx_out[qrow, :k_use] = cidx[order]
            dist_out[qrow, :k_use] = cdist[order]

    return idx_out, dist_out


def _resolve_ref_query_latent_keys(
    adata_ref: ad.AnnData,
    adata_query: ad.AnnData,
    *,
    ref_latent_key: str,
    query_latent_key: str,
) -> tuple[str, str]:
    ref_key_req = str(ref_latent_key)
    query_key_req = str(query_latent_key)

    if ref_key_req not in adata_ref.obsm:
        ref_alias = {
            "X_pca": "X_pca_harmony",
            "X_pca_harmony": "X_pca",
        }.get(ref_key_req, None)
        if ref_alias is not None and ref_alias in adata_ref.obsm:
            print(
                f"[ZMAP] ref latent '{ref_key_req}' not found; fallback to '{ref_alias}'."
            )
            ref_key_req = ref_alias
        else:
            raise KeyError(f"Missing pinned ref latent key in adata_ref.obsm: {ref_key_req}")

    if query_key_req not in adata_query.obsm:
        encoder_meta = adata_query.uns.get("zmap_encoder", {})
        enc_key = None
        if isinstance(encoder_meta, dict):
            k = encoder_meta.get("latent_out_key", None)
            if isinstance(k, str) and k in adata_query.obsm:
                enc_key = k
        if enc_key is not None:
            print(
                f"[ZMAP] query latent '{query_key_req}' not found; fallback to encoder output '{enc_key}'."
            )
            query_key_req = enc_key
        else:
            qry_alias = {
                "X_pca_pred": "X_pca_harmony_pred",
                "X_pca_harmony_pred": "X_pca_pred",
            }.get(query_key_req, None)
            if qry_alias is not None and qry_alias in adata_query.obsm:
                print(
                    f"[ZMAP] query latent '{query_key_req}' not found; fallback to '{qry_alias}'."
                )
                query_key_req = qry_alias
            else:
                raise KeyError(f"Missing query latent key in adata_query.obsm: {query_key_req}")
    return ref_key_req, query_key_req


def _resolve_query_latent_key_only(
    adata_query: ad.AnnData,
    *,
    query_latent_key: str,
) -> str:
    q_key = str(query_latent_key)
    if q_key in adata_query.obsm:
        return q_key

    encoder_meta = adata_query.uns.get("zmap_encoder", {})
    if isinstance(encoder_meta, dict):
        k = encoder_meta.get("latent_out_key", None)
        if isinstance(k, str) and k in adata_query.obsm:
            print(
                f"[ZMAP] query latent '{q_key}' not found; fallback to encoder output '{k}'."
            )
            return k

    alias = {
        "X_pca_pred": "X_pca_harmony_pred",
        "X_pca_harmony_pred": "X_pca_pred",
    }.get(q_key, None)
    if alias is not None and alias in adata_query.obsm:
        print(f"[ZMAP] query latent '{q_key}' not found; fallback to '{alias}'.")
        return alias
    raise KeyError(f"Missing query latent key in adata_query.obsm: {q_key}")


def _make_faiss_cache_prefix(
    *,
    ref_source: str,
    ref_latent_key: str,
    n_ref: int,
    knn_metric: str,
    knn_l2norm: bool,
) -> str:
    return (
        f"{str(ref_source)}|{str(ref_latent_key)}|"
        f"n_ref={int(n_ref)}|metric={str(knn_metric)}|l2={int(bool(knn_l2norm))}"
    )


def _assign_pseudo_tissue_from_arrays(
    adata_query: ad.AnnData,
    *,
    query_latent: np.ndarray,
    ref_latent: np.ndarray,
    ref_tissue: np.ndarray,
    tissue_col: str,
    pseudo_col: str,
    k: int,
    threshold: float,
    margin_threshold: float,
    knn_metric: str,
    knn_backend: str,
    knn_device: str,
    knn_nprobe: int | None,
    faiss_cache_key: str | None,
    write_to_tissue_col: bool,
    unknown_label: str = "unknown",
    source: str = "manual",
) -> np.ndarray:
    if ref_latent.shape[0] <= 0:
        raise ValueError("No reference rows available for pseudo tissue.")
    k_use = min(int(ref_latent.shape[0]), max(1, int(k)))
    idx_t, dist_t = _knn_global(
        ref_latent,
        query_latent,
        k=int(k_use),
        metric=knn_metric,
        knn_backend=knn_backend,
        knn_device=knn_device,
        knn_nprobe=knn_nprobe,
        faiss_cache_key=faiss_cache_key,
    )
    pseudo_pred, pseudo_prob, pseudo_margin, pseudo_entropy, _ = _knn_vote(
        np.asarray(ref_tissue, dtype=object),
        idx_t,
        dist_t,
    )

    adata_query.obs[pseudo_col] = pseudo_pred.astype(str)
    adata_query.obs[f"{pseudo_col}_max_prob"] = pseudo_prob
    adata_query.obs[f"{pseudo_col}_margin"] = pseudo_margin
    adata_query.obs[f"{pseudo_col}_entropy"] = pseudo_entropy
    # Compatibility alias for generic conf-entropy-margin plotting utility.
    adata_query.obs[f"{pseudo_col}_knn_max_prob"] = pseudo_prob
    adata_query.obs[f"{pseudo_col}_knn_margin"] = pseudo_margin
    adata_query.obs[f"{pseudo_col}_knn_entropy"] = pseudo_entropy

    thr = float(threshold)
    mar_thr = float(margin_threshold)
    query_tissue = np.full(query_latent.shape[0], str(unknown_label), dtype=object)
    keep_hi = np.ones(query_latent.shape[0], dtype=bool)
    if thr > 0:
        keep_hi &= pseudo_prob >= thr
    if mar_thr > 0:
        keep_hi &= pseudo_margin >= mar_thr
    query_tissue[keep_hi] = pseudo_pred[keep_hi].astype(str)
    if write_to_tissue_col:
        adata_query.obs[tissue_col] = query_tissue.astype(str)

    adata_query.uns.setdefault("zmap_pseudo_tissue", {})
    adata_query.uns["zmap_pseudo_tissue"] = {
        "enabled": True,
        "source": str(source),
        "tissue_col": str(tissue_col),
        "pseudo_col": str(pseudo_col),
        "k": int(k_use),
        "threshold": float(thr),
        "margin_threshold": float(mar_thr),
        "knn_metric": str(knn_metric),
        "knn_backend": str(knn_backend),
        "knn_device": str(knn_device),
        "knn_nprobe": (None if knn_nprobe is None else int(knn_nprobe)),
        "write_to_tissue_col": bool(write_to_tissue_col),
        "unknown_label": str(unknown_label),
    }
    return query_tissue


def _prepare_knn_context(
    adata_query: ad.AnnData,
    adata_ref: ad.AnnData | None,
    *,
    ref_latent_key: str,
    query_latent_key: str,
    label_col: str,
    tissue_col: str,
    k: int,
    tissue_mode: str,
    include_unassigned: bool,
    knn_l2norm: bool,
    knn_metric: str,
    knn_backend: str,
    knn_device: str,
    knn_nprobe: int | None,
) -> dict[str, object]:
    ref_source = "input_adata_ref"
    if adata_ref is None:
        adata_ref, ref_source = _load_pinned_reference(use_cache=True)

    ref_key_req, query_key_req = _resolve_ref_query_latent_keys(
        adata_ref,
        adata_query,
        ref_latent_key=ref_latent_key,
        query_latent_key=query_latent_key,
    )
    if label_col not in adata_ref.obs:
        raise KeyError(f"Missing label column in adata_ref.obs: {label_col}")
    if tissue_mode not in {"none", "hard", "soft"}:
        raise ValueError("tissue_mode must be one of {'none', 'hard', 'soft'}.")

    ref_latent_all = np.asarray(adata_ref.obsm[ref_key_req], dtype=np.float32)
    query_latent = np.asarray(adata_query.obsm[query_key_req], dtype=np.float32)
    ref_labels_all = adata_ref.obs[label_col].astype(str).to_numpy()

    if include_unassigned:
        keep = np.ones(ref_labels_all.shape[0], dtype=bool)
    else:
        keep = ref_labels_all != "unassigned"
    ref_latent = ref_latent_all[keep]
    ref_labels = ref_labels_all[keep]
    if ref_latent.shape[0] < int(k):
        raise ValueError(f"Not enough reference cells for k={k}: {ref_latent.shape[0]}")

    ref_use = ref_latent
    query_use = query_latent
    if knn_l2norm:
        ref_use = _l2_normalize(ref_use)
        query_use = _l2_normalize(query_use)
    faiss_cache_prefix = _make_faiss_cache_prefix(
        ref_source=ref_source,
        ref_latent_key=ref_key_req,
        n_ref=int(ref_use.shape[0]),
        knn_metric=knn_metric,
        knn_l2norm=bool(knn_l2norm),
    )

    ref_tissue = None
    query_tissue = None
    mode = tissue_mode
    if mode != "none":
        if tissue_col in adata_ref.obs and tissue_col in adata_query.obs:
            ref_tissue = adata_ref.obs[tissue_col].astype(str).to_numpy()[keep]
            query_tissue = adata_query.obs[tissue_col].astype(str).to_numpy()
        else:
            if tissue_col not in adata_ref.obs:
                raise KeyError(
                    f"Missing tissue column in adata_ref.obs: {tissue_col} "
                    "(required for tissue_mode='hard'/'soft')."
                )
            if tissue_col not in adata_query.obs:
                raise KeyError(
                    f"Missing tissue column in adata_query.obs: {tissue_col}. "
                    "Run zmap.predict_pseudo_tissue(...) first, or set tissue_mode='none'."
                )
            raise ValueError(
                "Hard/soft tissue mode requires tissue annotations in both ref and query."
            )

    return {
        "adata_ref": adata_ref,
        "ref_source": ref_source,
        "ref_latent_key": ref_key_req,
        "query_latent_key": query_key_req,
        "label_col": label_col,
        "tissue_col": tissue_col_use,
        "mode": mode,
        "keep": keep,
        "ref_labels": ref_labels,
        "ref_tissue": ref_tissue,
        "query_tissue": query_tissue,
        "ref_use": ref_use,
        "query_use": query_use,
        "faiss_cache_prefix": faiss_cache_prefix,
    }


def _knn_cache_config(
    *,
    ref_source: str,
    ref_latent_key: str,
    query_latent_key: str,
    mode: str,
    k: int,
    metric: str,
    knn_backend: str,
    knn_device: str,
    knn_nprobe: int | None,
    knn_l2norm: bool,
    tissue_penalty_lambda: float,
    n_ref: int,
    n_query: int,
) -> dict[str, object]:
    return {
        "ref_source": str(ref_source),
        "ref_latent_key": str(ref_latent_key),
        "query_latent_key": str(query_latent_key),
        "mode": str(mode),
        "k": int(k),
        "metric": str(metric),
        "knn_backend": str(knn_backend),
        "knn_device": str(knn_device),
        "knn_nprobe": (None if knn_nprobe is None else int(knn_nprobe)),
        "knn_l2norm": bool(knn_l2norm),
        "tissue_penalty_lambda": float(tissue_penalty_lambda),
        "n_ref": int(n_ref),
        "n_query": int(n_query),
    }


def _get_or_compute_knn_neighbors(
    adata_query: ad.AnnData,
    *,
    adata_ref: ad.AnnData,
    keep: np.ndarray,
    ref_source: str,
    ref_latent_key: str,
    query_latent_key: str,
    mode: str,
    k: int,
    metric: str,
    knn_backend: str,
    knn_device: str,
    knn_nprobe: int | None,
    knn_l2norm: bool,
    tissue_penalty_lambda: float,
    faiss_cache_prefix: str | None,
    ref_use: np.ndarray,
    query_use: np.ndarray,
    ref_tissue: np.ndarray | None,
    query_tissue: np.ndarray | None,
    reuse_knn_cache: bool,
) -> tuple[np.ndarray, np.ndarray, bool, dict[str, object]]:
    cfg = _knn_cache_config(
        ref_source=ref_source,
        ref_latent_key=ref_latent_key,
        query_latent_key=query_latent_key,
        mode=mode,
        k=int(k),
        metric=metric,
        knn_backend=knn_backend,
        knn_device=knn_device,
        knn_nprobe=knn_nprobe,
        knn_l2norm=knn_l2norm,
        tissue_penalty_lambda=tissue_penalty_lambda,
        n_ref=int(ref_use.shape[0]),
        n_query=int(query_use.shape[0]),
    )
    cache = adata_query.uns.get("zmap_knn_cache", None)
    if (
        reuse_knn_cache
        and isinstance(cache, dict)
        and cache.get("config") == cfg
        and "idx" in cache
        and "dist" in cache
    ):
        idx = np.asarray(cache["idx"], dtype=np.int64)
        dist = np.asarray(cache["dist"], dtype=np.float32)
        if idx.shape == (query_use.shape[0], int(k)) and dist.shape == (query_use.shape[0], int(k)):
            return idx, dist, True, cfg

    if mode == "none":
        idx, dist = _knn_global(
            ref_use,
            query_use,
            k=int(k),
            metric=metric,
            knn_backend=knn_backend,
            knn_device=knn_device,
            knn_nprobe=knn_nprobe,
            faiss_cache_key=(
                None
                if faiss_cache_prefix is None
                else f"{str(faiss_cache_prefix)}|global"
            ),
        )
    elif mode == "hard":
        if ref_tissue is None or query_tissue is None:
            raise ValueError("Hard tissue mode requires tissue annotations in both ref and query.")
        idx, dist = _knn_hard_tissue(
            ref_use,
            ref_tissue,
            query_use,
            query_tissue,
            k=int(k),
            metric=metric,
            knn_backend=knn_backend,
            knn_device=knn_device,
            knn_nprobe=knn_nprobe,
            faiss_cache_prefix=(
                None
                if faiss_cache_prefix is None
                else f"{str(faiss_cache_prefix)}|hard"
            ),
        )
    else:
        if ref_tissue is None or query_tissue is None:
            raise ValueError("Soft tissue mode requires tissue annotations in both ref and query.")
        idx, dist = _knn_soft_tissue(
            ref_use,
            ref_tissue,
            query_use,
            query_tissue,
            k=int(k),
            metric=metric,
            penalty_lambda=float(tissue_penalty_lambda),
            knn_backend=knn_backend,
            knn_device=knn_device,
            knn_nprobe=knn_nprobe,
            faiss_cache_prefix=(
                None
                if faiss_cache_prefix is None
                else f"{str(faiss_cache_prefix)}|soft"
            ),
        )
    idx = idx[:, : int(k)]
    dist = dist[:, : int(k)]

    adata_query.uns["zmap_knn_cache"] = {
        "config": cfg,
        "idx": idx,
        "dist": dist,
    }
    return idx, dist, False, cfg


def predict_pseudo_tissue(
    adata_query: ad.AnnData,
    adata_ref: ad.AnnData | None = None,
    *,
    ref_latent_key: str = DEFAULT_REF_LATENT_KEY,
    query_latent_key: str = "X_pca_pred",
    label_col: str = DEFAULT_LABEL_COL,
    tissue_col: str = DEFAULT_TISSUE_COL,
    k: int = 15,
    pseudo_tissue_k: int | None = None,
    pseudo_tissue_threshold: float = 0.0,
    pseudo_tissue_margin_threshold: float = 0.0,
    unknown_label: str = "unknown",
    include_unassigned: bool = False,
    knn_metric: str = "euclidean",
    knn_backend: str = "auto",
    knn_device: str = "auto",
    knn_nprobe: int | None = None,
    knn_l2norm: bool = False,
    pseudo_col: str | None = None,
    write_to_tissue_col: bool = True,
) -> ad.AnnData:
    """
    Predict pseudo tissue labels for query and optionally write them to `tissue_col`.

    This function enables a two-step workflow:
      1) `predict_pseudo_tissue(...)`
      2) `predict_labels_kNN(..., tissue_aware=True, tissue_col=...)`
    """
    ref_source = "input_adata_ref"
    if adata_ref is None:
        adata_ref, ref_source = _load_pinned_reference(use_cache=True)

    if tissue_col not in adata_ref.obs:
        raise KeyError(f"Missing tissue column in adata_ref.obs: {tissue_col}")

    ref_key_req, query_key_req = _resolve_ref_query_latent_keys(
        adata_ref,
        adata_query,
        ref_latent_key=ref_latent_key,
        query_latent_key=query_latent_key,
    )
    ref_latent_all = np.asarray(adata_ref.obsm[ref_key_req], dtype=np.float32)
    query_latent = np.asarray(adata_query.obsm[query_key_req], dtype=np.float32)

    keep = np.ones(adata_ref.n_obs, dtype=bool)
    if label_col in adata_ref.obs and (not bool(include_unassigned)):
        keep &= adata_ref.obs[label_col].astype(str).to_numpy() != "unassigned"
    ref_tissue_all = adata_ref.obs[tissue_col].astype(str).to_numpy()
    keep &= ref_tissue_all != "unassigned"
    keep &= ~pd.isna(adata_ref.obs[tissue_col]).to_numpy()

    ref_latent = ref_latent_all[keep]
    ref_tissue = ref_tissue_all[keep]
    if ref_latent.shape[0] <= 0:
        raise ValueError("No valid reference rows left for pseudo tissue prediction.")

    ref_use = ref_latent
    query_use = query_latent
    if bool(knn_l2norm):
        ref_use = _l2_normalize(ref_use)
        query_use = _l2_normalize(query_use)
    faiss_cache_prefix = _make_faiss_cache_prefix(
        ref_source=ref_source,
        ref_latent_key=ref_key_req,
        n_ref=int(ref_use.shape[0]),
        knn_metric=knn_metric,
        knn_l2norm=bool(knn_l2norm),
    )

    if pseudo_tissue_k is None:
        pseudo_k = min(ref_use.shape[0], max(int(k), 31))
    else:
        pseudo_k = min(ref_use.shape[0], max(1, int(pseudo_tissue_k)))

    pseudo_col_use = str(pseudo_col) if pseudo_col is not None else f"{tissue_col}_pseudo"
    _assign_pseudo_tissue_from_arrays(
        adata_query,
        query_latent=query_use,
        ref_latent=ref_use,
        ref_tissue=np.asarray(ref_tissue, dtype=object),
        tissue_col=str(tissue_col),
        pseudo_col=str(pseudo_col_use),
        k=int(pseudo_k),
        threshold=float(pseudo_tissue_threshold),
        margin_threshold=float(pseudo_tissue_margin_threshold),
        knn_metric=knn_metric,
        knn_backend=knn_backend,
        knn_device=knn_device,
        knn_nprobe=knn_nprobe,
        faiss_cache_key=f"{str(faiss_cache_prefix)}|global",
        write_to_tissue_col=bool(write_to_tissue_col),
        unknown_label=str(unknown_label),
        source="manual_predict_pseudo_tissue",
    )
    adata_query.uns.setdefault("zmap_pseudo_tissue", {})
    adata_query.uns["zmap_pseudo_tissue"].update(
        {
            "ref_source": str(ref_source),
            "ref_latent_key": str(ref_key_req),
            "query_latent_key": str(query_key_req),
            "label_col_filter": str(label_col),
            "include_unassigned_ref": bool(include_unassigned),
            "n_ref_used": int(ref_use.shape[0]),
            "n_query": int(query_use.shape[0]),
            "faiss_cache_prefix": str(faiss_cache_prefix),
        }
    )
    print(
        "[ZMAP] pseudo tissue done: "
        f"col='{pseudo_col_use}', k={int(pseudo_k)}, "
        f"threshold={float(pseudo_tissue_threshold):.3f}, "
        f"margin_threshold={float(pseudo_tissue_margin_threshold):.3f}"
    )
    return adata_query


def time_prediction(
    adata_query: ad.AnnData,
    adata_ref: ad.AnnData | None = None,
    *,
    ref_latent_key: str = DEFAULT_REF_LATENT_KEY,
    query_latent_key: str = "X_pca_pred",
    k: int = 15,
    label_col: str = DEFAULT_LABEL_COL,
    tissue_aware: bool = False,
    tissue_mode: str = "hard",
    tissue_col: str | None = DEFAULT_TISSUE_COL,
    predict_tissue_level: bool = False,
    tissue_penalty_lambda: float = 1.0,
    knn_metric: str = "euclidean",
    knn_backend: str = "auto",
    knn_device: str = "auto",
    knn_nprobe: int | None = None,
    knn_l2norm: bool = False,
    class_prior_alpha: float = 0.0,
    pseudo_tissue_k: int | None = None,
    pseudo_tissue_threshold: float = 0.0,
    pseudo_tissue_margin_threshold: float = 0.0,
    reuse_knn_cache: bool = True,
    include_unassigned: bool = False,
    time_col: str = "time_group_id",
    time_order: str | list[str] | None = None,
    time_topk: int = 5,
    time_hard_topk: int = 5,
    time_trim_extremes: int = 1,
    time_tau: float = 0.0,
    time_monotone_delta: int = 0,
    time_monotone_gamma: float = 1.0,
) -> ad.AnnData:
    """
    Run kNN-based time prediction on query latent.

    This function reuses cached neighbors from `adata_query.uns['zmap_knn_cache']`
    when latent/tissue/kNN settings are unchanged.
    """
    tissue_aware_resolved, mode_requested = _resolve_tissue_aware_mode(
        tissue_aware=bool(tissue_aware),
        tissue_mode=tissue_mode,
    )
    tissue_col_use = str(tissue_col or DEFAULT_TISSUE_COL)
    if tissue_aware_resolved and tissue_col is None and not bool(predict_tissue_level):
        raise ValueError(
            "tissue_aware=True requires tissue_col, or set predict_tissue_level=True "
            "to infer query tissue labels before tissue-aware kNN."
        )
    if tissue_aware_resolved and bool(predict_tissue_level):
        adata_query = predict_pseudo_tissue(
            adata_query,
            adata_ref,
            ref_latent_key=ref_latent_key,
            query_latent_key=query_latent_key,
            label_col=label_col,
            tissue_col=tissue_col_use,
            k=int(k),
            pseudo_tissue_k=pseudo_tissue_k,
            pseudo_tissue_threshold=float(pseudo_tissue_threshold),
            pseudo_tissue_margin_threshold=float(pseudo_tissue_margin_threshold),
            include_unassigned=include_unassigned,
            knn_metric=knn_metric,
            knn_backend=knn_backend,
            knn_device=knn_device,
            knn_nprobe=knn_nprobe,
            knn_l2norm=knn_l2norm,
            write_to_tissue_col=True,
        )
    elif (pseudo_tissue_k is not None) or (float(pseudo_tissue_threshold) > 0):
        print(
            "[ZMAP] time_prediction: pseudo_tissue_k / pseudo_tissue_threshold are used "
            "only when predict_tissue_level=True."
        )
    ctx = _prepare_knn_context(
        adata_query,
        adata_ref,
        ref_latent_key=ref_latent_key,
        query_latent_key=query_latent_key,
        label_col=label_col,
        tissue_col=tissue_col_use,
        k=int(k),
        tissue_mode=mode_requested,
        include_unassigned=include_unassigned,
        knn_l2norm=knn_l2norm,
        knn_metric=knn_metric,
        knn_backend=knn_backend,
        knn_device=knn_device,
        knn_nprobe=knn_nprobe,
    )
    adata_ref = ctx["adata_ref"]  # type: ignore[assignment]
    ref_source = str(ctx["ref_source"])
    ref_latent_key = str(ctx["ref_latent_key"])
    query_latent_key = str(ctx.get("query_latent_key", query_latent_key))
    mode = str(ctx["mode"])
    keep = np.asarray(ctx["keep"], dtype=bool)
    ref_tissue = ctx["ref_tissue"]
    query_tissue = ctx["query_tissue"]
    ref_use = np.asarray(ctx["ref_use"], dtype=np.float32)
    query_use = np.asarray(ctx["query_use"], dtype=np.float32)
    faiss_cache_prefix = ctx.get("faiss_cache_prefix", None)

    idx, dist, cache_reused, _ = _get_or_compute_knn_neighbors(
        adata_query,
        adata_ref=adata_ref,
        keep=keep,
        ref_source=ref_source,
        ref_latent_key=ref_latent_key,
        query_latent_key=query_latent_key,
        mode=mode,
        k=int(k),
        metric=knn_metric,
        knn_backend=knn_backend,
        knn_device=knn_device,
        knn_nprobe=knn_nprobe,
        knn_l2norm=knn_l2norm,
        tissue_penalty_lambda=float(tissue_penalty_lambda),
        faiss_cache_prefix=(None if faiss_cache_prefix is None else str(faiss_cache_prefix)),
        ref_use=ref_use,
        query_use=query_use,
        ref_tissue=None if ref_tissue is None else np.asarray(ref_tissue, dtype=object),
        query_tissue=None if query_tissue is None else np.asarray(query_tissue, dtype=object),
        reuse_knn_cache=bool(reuse_knn_cache),
    )

    summary = _compute_time_prediction_from_knn(
        adata_query,
        adata_ref,
        keep=keep,
        idx=idx,
        dist=dist,
        time_col=time_col,
        time_order=time_order,
        time_topk=int(time_topk),
        time_hard_topk=int(time_hard_topk),
        time_trim_extremes=int(time_trim_extremes),
        time_tau=float(time_tau),
        time_monotone_delta=int(time_monotone_delta),
        time_monotone_gamma=float(time_monotone_gamma),
        output_prefix=str(time_col),
    )
    summary["knn_cache_reused"] = bool(cache_reused)
    summary["knn_backend"] = str(knn_backend)
    summary["knn_device"] = str(knn_device)
    summary["knn_nprobe"] = (None if knn_nprobe is None else int(knn_nprobe))
    summary["class_prior_alpha"] = float(class_prior_alpha)
    summary["pseudo_tissue_k"] = (None if pseudo_tissue_k is None else int(pseudo_tissue_k))
    summary["pseudo_tissue_threshold"] = float(pseudo_tissue_threshold)
    adata_query.uns["zmap_time_prediction"] = summary

    teval = summary.get("evaluation", {})
    if isinstance(teval, dict) and int(teval.get("n_valid", 0)) > 0:
        print(
            "[ZMAP] time_prediction "
            f"exact={float(teval.get('exact_accuracy', np.nan)):.6f} "
            f"ordinal_pm1={float(teval.get('ordinal_accuracy_pm1', np.nan)):.6f} "
            f"mae={float(teval.get('mean_abs_error', np.nan)):.6f} "
            f"(n={int(teval.get('n_valid', 0))})"
        )
    else:
        print(f"[ZMAP] time_prediction finished (time_col='{time_col}').")
    return adata_query


def plot_confusion_matrix(
    adata_query: ad.AnnData,
    *,
    source: str = "knn",
    level: str | None = None,
    normalize: bool = False,
    save_path: str | None = None,
    show: bool = True,
    dpi: int = 200,
) -> None:
    """
    Plot confusion matrix using the same visual style as legacy train/apply scripts.

    source:
      - "knn": read from adata_query.uns["zmap_knn"]["evaluation"]
      - "time": read from adata_query.uns["zmap_time_prediction"]["confusion_matrix"]
    level:
      - for source="knn", pick one of:
        "ZMAP_GermLayer", "ZMAP_Tissue", "ZMAP_CellType", "ZMAP_CellTypeFine"
      - None uses the primary confusion matrix under evaluation["confusion_matrix"].
    """
    src = str(source).lower().strip()
    if src not in {"knn", "time"}:
        raise ValueError("source must be 'knn' or 'time'.")

    cm_obj: dict[str, object] | None = None
    if src == "knn":
        ev = adata_query.uns.get("zmap_knn", {}).get("evaluation", {})
        if not isinstance(ev, dict):
            raise KeyError("Missing zmap_knn evaluation results.")
        if level is None:
            cm_obj = ev.get("confusion_matrix", None)  # type: ignore[assignment]
        else:
            by_level = ev.get("confusion_by_level", {})
            if not isinstance(by_level, dict):
                raise KeyError("Missing confusion_by_level in knn evaluation.")
            cm_obj = by_level.get(level, None)  # type: ignore[assignment]
    else:
        tp = adata_query.uns.get("zmap_time_prediction", {})
        if not isinstance(tp, dict):
            raise KeyError("Missing zmap_time_prediction results.")
        cm_obj = tp.get("confusion_matrix", None)  # type: ignore[assignment]

    if not isinstance(cm_obj, dict):
        where = f"source='{src}'" + (f", level='{level}'" if level is not None else "")
        raise KeyError(f"Confusion matrix not found for {where}.")

    labels = cm_obj.get("labels", None)
    matrix = cm_obj.get("matrix", None)
    if labels is None or matrix is None:
        raise ValueError("Confusion matrix object must contain 'labels' and 'matrix'.")

    labels_arr = [str(x) for x in labels]  # type: ignore[arg-type]
    cm = np.asarray(matrix, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got shape={cm.shape}.")
    if cm.shape[0] != len(labels_arr):
        raise ValueError(
            "Confusion matrix size mismatch: "
            f"{cm.shape[0]}x{cm.shape[1]} vs labels={len(labels_arr)}"
        )

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib unavailable: {e}") from e

    fig_size = max(8.0, min(30.0, len(labels_arr) * 0.35))
    if normalize:
        col_sums = cm.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = np.where(col_sums > 0, (cm / col_sums), 0.0)
    else:
        col_sums = None
        cm_plot = cm.astype(float)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_plot, aspect="auto", cmap="Blues")
    if cm_plot.size:
        row_max_idx = np.argmax(cm_plot, axis=1)
        for i, j in enumerate(row_max_idx):
            if normalize and col_sums is not None and col_sums[0, j] == 0:
                continue
            ax.text(
                j,
                i,
                f"{cm_plot[i, j]:.2f}" if normalize else f"{int(cm_plot[i, j])}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_title("")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if len(labels_arr) <= 120:
        ticks = np.arange(len(labels_arr))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels_arr, fontsize=6, rotation=90)
        ax.set_yticklabels(labels_arr, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=int(dpi))
    if show:
        plt.show()
    plt.close(fig)


def plot_leiden_celltype(
    adata_query: ad.AnnData,
    *,
    leiden_col: str = "leiden",
    pred_col: str = "ZMAP_CellType_pred",
    unknown_label: str = "unassign",
    top_labels: int = 20,
    drop_unknown_and_renorm: bool = False,
    cmap: str = "Blues",
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    show: bool = True,
    dpi: int = 300,
) -> dict[str, object]:
    """
    Plot Leiden x predicted-celltype heatmap from adata_query.obs.

    Returns a dict with count/proportion tables used for plotting.
    """
    if leiden_col not in adata_query.obs:
        raise KeyError(f"Missing leiden_col in adata_query.obs: {leiden_col}")
    if pred_col not in adata_query.obs:
        raise KeyError(f"Missing pred_col in adata_query.obs: {pred_col}")
    if top_labels <= 0:
        raise ValueError("top_labels must be positive.")

    obs = adata_query.obs[[leiden_col, pred_col]].copy()
    obs[leiden_col] = obs[leiden_col].astype(str)
    obs[pred_col] = obs[pred_col].astype(str)

    ct = pd.crosstab(obs[leiden_col], obs[pred_col])
    def _leiden_sort_key(x: object) -> tuple[object, ...]:
        s = str(x)
        try:
            return (0, float(s), s)
        except Exception:
            m = re.search(r"[-+]?\d*\.?\d+", s)
            if m is not None:
                try:
                    return (0, float(m.group()), s)
                except Exception:
                    pass
            return (1, s)

    ordered_rows = sorted(list(ct.index), key=_leiden_sort_key)
    ct = ct.reindex(ordered_rows)
    prop = ct.div(ct.sum(axis=1), axis=0).fillna(0.0)

    if drop_unknown_and_renorm:
        ct_plot = ct.drop(columns=[unknown_label], errors="ignore")
        prop_plot = ct_plot.div(ct_plot.sum(axis=1), axis=0).fillna(0.0)
    else:
        ct_plot = ct
        prop_plot = prop

    if prop_plot.shape[1] == 0:
        raise ValueError("No label columns to plot after filtering.")

    top = ct_plot.sum(axis=0).sort_values(ascending=False).head(int(top_labels)).index
    heat = prop_plot[top]

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        raise RuntimeError(f"plot_leiden_celltype requires matplotlib+seaborn: {e}") from e

    if figsize is None:
        fig_h = max(8.0, 0.28 * heat.shape[0])
        fig_w = max(10.0, 0.45 * heat.shape[1])
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(heat, cmap=cmap, yticklabels=1, ax=ax)
    ax.set_title("")
    ax.set_xlabel(pred_col)
    ax.set_ylabel(leiden_col)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=int(dpi))
    if show:
        plt.show()
    plt.close(fig)

    return {
        "counts": ct,
        "proportions": prop,
        "counts_plot": ct_plot,
        "proportions_plot": prop_plot,
        "heatmap_table": heat,
        "top_labels": [str(x) for x in top],
        "drop_unknown_and_renorm": bool(drop_unknown_and_renorm),
    }


def report_leiden_celltype_topk(
    adata_query: ad.AnnData,
    *,
    leiden_col: str = "leiden",
    pred_col: str = "ZMAP_CellType_pred",
    topk: int = 5,
    unknown_label: str = "unassigned",
    drop_unknown: bool = True,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Report top-k predicted celltypes for each Leiden cluster.

    Returns a long-format table with columns:
      [leiden, rank, celltype, count, proportion]
    """
    if leiden_col not in adata_query.obs:
        raise KeyError(f"Missing leiden_col in adata_query.obs: {leiden_col}")
    if pred_col not in adata_query.obs:
        raise KeyError(f"Missing pred_col in adata_query.obs: {pred_col}")
    if int(topk) <= 0:
        raise ValueError("topk must be positive.")

    obs = adata_query.obs[[leiden_col, pred_col]].copy()
    obs[leiden_col] = obs[leiden_col].astype(str)
    obs[pred_col] = obs[pred_col].astype(str)

    ct = pd.crosstab(obs[leiden_col], obs[pred_col])
    if bool(drop_unknown):
        ct = ct.drop(columns=[unknown_label], errors="ignore")
    if ct.shape[1] == 0:
        raise ValueError("No label columns to summarize after filtering.")

    def _leiden_sort_key(x: object) -> tuple[object, ...]:
        s = str(x)
        try:
            return (0, float(s), s)
        except Exception:
            m = re.search(r"[-+]?\d*\.?\d+", s)
            if m is not None:
                try:
                    return (0, float(m.group()), s)
                except Exception:
                    pass
            return (1, s)

    ct = ct.reindex(sorted(list(ct.index), key=_leiden_sort_key))
    totals = ct.sum(axis=1).astype(float)

    rows: list[dict[str, object]] = []
    for leiden in ct.index:
        row = ct.loc[leiden].sort_values(ascending=False).head(int(topk))
        denom = float(totals.loc[leiden])
        for rank, (lbl, cnt) in enumerate(row.items(), start=1):
            count_i = int(cnt)
            prop = (float(count_i) / denom) if denom > 0 else 0.0
            rows.append(
                {
                    "leiden": str(leiden),
                    "rank": int(rank),
                    "celltype": str(lbl),
                    "count": count_i,
                    "proportion": float(prop),
                }
            )

    out = pd.DataFrame(rows)
    if save_path:
        out.to_csv(str(save_path), index=False)
    return out


def plot_conf_entropy_margin(
    adata_query: ad.AnnData,
    *,
    base: str = "ZMAP_CellType",
    threshold: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cmap: str = "viridis",
    figsize: tuple[float, float] = (6.0, 5.0),
    point_size: float = 6.0,
    alpha: float = 0.7,
    fail_use_margin_color: bool = True,
    fail_color: str = "black",
    fail_alpha: float = 0.6,
    save_path: str | None = None,
    show: bool = True,
    dpi: int = 300,
) -> dict[str, object]:
    """
    Scatter plot for confidence vs entropy colored by margin.

    threshold:
      (conf_threshold, entropy_threshold, margin_threshold)
      - conf_threshold > 0: keep points with conf > threshold
      - entropy_threshold > 0: keep points with entropy < threshold
      - margin_threshold > 0: keep points with margin > threshold
      Points failing any active threshold are drawn as 'X'.
    """
    if len(threshold) != 3:
        raise ValueError("threshold must be a length-3 tuple: (conf, entropy, margin).")
    conf_thr, ent_thr, mar_thr = [float(x) for x in threshold]

    x_col = f"{base}_knn_max_prob"
    y_col = f"{base}_knn_entropy"
    c_col = f"{base}_knn_margin"
    for col in (x_col, y_col, c_col):
        if col not in adata_query.obs:
            raise KeyError(f"Missing column in adata_query.obs: {col}")

    df = adata_query.obs[[x_col, y_col, c_col]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.shape[0] == 0:
        raise ValueError("No valid rows after dropping NaN/inf in conf/entropy/margin columns.")

    keep_mask = np.ones(df.shape[0], dtype=bool)
    if conf_thr > 0:
        keep_mask &= df[x_col].to_numpy() > conf_thr
    if ent_thr > 0:
        keep_mask &= df[y_col].to_numpy() < ent_thr
    if mar_thr > 0:
        keep_mask &= df[c_col].to_numpy() > mar_thr
    fail_mask = ~keep_mask

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"plot_conf_entropy_margin requires matplotlib: {e}") from e

    fig, ax = plt.subplots(figsize=figsize)
    vmin = float(np.nanmin(df[c_col].to_numpy()))
    vmax = float(np.nanmax(df[c_col].to_numpy()))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-8
    if np.any(keep_mask):
        sc = ax.scatter(
            df.loc[keep_mask, x_col],
            df.loc[keep_mask, y_col],
            c=df.loc[keep_mask, c_col],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=float(point_size),
            alpha=float(alpha),
            linewidths=0,
            rasterized=True,
        )
    else:
        # Keep colorbar available even when all points fail threshold.
        sc = ax.scatter(
            df[x_col],
            df[y_col],
            c=df[c_col],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=0.0,
            alpha=0.0,
            linewidths=0,
            rasterized=True,
        )
    if np.any(fail_mask):
        if bool(fail_use_margin_color):
            ax.scatter(
                df.loc[fail_mask, x_col],
                df.loc[fail_mask, y_col],
                marker="x",
                c=df.loc[fail_mask, c_col],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=max(12.0, float(point_size) * 2.0),
                alpha=float(fail_alpha),
                linewidths=0.8,
                rasterized=True,
            )
        else:
            ax.scatter(
                df.loc[fail_mask, x_col],
                df.loc[fail_mask, y_col],
                marker="x",
                c=fail_color,
                s=max(12.0, float(point_size) * 2.0),
                alpha=float(fail_alpha),
                linewidths=0.8,
                rasterized=True,
            )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Entropy")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Margin")
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=int(dpi))
    if show:
        plt.show()
    plt.close(fig)

    return {
        "x_col": x_col,
        "y_col": y_col,
        "c_col": c_col,
        "threshold": (conf_thr, ent_thr, mar_thr),
        "n_total": int(df.shape[0]),
        "n_keep": int(np.sum(keep_mask)),
        "n_fail": int(np.sum(fail_mask)),
    }


def plot_pseudo_tissue_conf_entropy_margin(
    adata_query: ad.AnnData,
    *,
    pseudo_base: str = "ZMAP_Tissue_pseudo",
    threshold: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cmap: str = "viridis",
    figsize: tuple[float, float] = (6.0, 5.0),
    point_size: float = 6.0,
    alpha: float = 0.7,
    fail_use_margin_color: bool = True,
    fail_color: str = "black",
    fail_alpha: float = 0.6,
    save_path: str | None = None,
    show: bool = True,
    dpi: int = 300,
) -> dict[str, object]:
    """
    Convenience wrapper of `plot_conf_entropy_margin` for pseudo tissue scores.
    """
    return plot_conf_entropy_margin(
        adata_query,
        base=str(pseudo_base),
        threshold=threshold,
        cmap=cmap,
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
        fail_use_margin_color=fail_use_margin_color,
        fail_color=fail_color,
        fail_alpha=fail_alpha,
        save_path=save_path,
        show=show,
        dpi=dpi,
    )


def plot_projection_celltype_overlay(
    adata_query: ad.AnnData,
    *,
    adata_ref: ad.AnnData | None = None,
    query_umap_key: str = "X_umap_proj",
    ref_umap_key: str = "X_umap",
    query_label_col: str = "ZMAP_CellType_pred",
    color_map: dict[str, str] | None = None,
    color_uns_key: str | None = None,
    default_color: str = "#808080",
    ref_color: str = "lightgray",
    ref_point_size: float = 0.5,
    ref_alpha: float = 0.3,
    query_point_size: float = 15.0,
    query_alpha: float = 0.8,
    label_fontsize: float = 10.0,
    label_outline_width: float = 3.0,
    label_weight: str = "normal",
    use_adjust_text: bool = True,
    adjust_arrowprops: dict[str, object] | None = None,
    adjust_expand_points: tuple[float, float] = (1.5, 1.5),
    adjust_expand_text: tuple[float, float] = (1.2, 1.2),
    adjust_force_points: tuple[float, float] = (0.5, 0.5),
    adjust_force_text: tuple[float, float] = (0.7, 0.7),
    figsize: tuple[float, float] = (20.0, 16.0),
    dpi: int = 300,
    rasterized: bool = True,
    title: str | None = None,
    remove_axes: bool = True,
    save_path: str | None = None,
    show: bool = True,
) -> dict[str, object]:
    """
    Plot query-on-reference UMAP overlay using fixed coordinates in obsm.

    Style is aligned with the provided standalone script:
      - reference in light gray background
      - query points colored by cell type
      - centroid labels with white stroke
      - optional adjustText label repulsion with thin connector lines
    """
    if adata_ref is None:
        adata_ref, _ = _load_pinned_reference(use_cache=True)

    if query_umap_key not in adata_query.obsm:
        raise KeyError(f"Missing query_umap_key in adata_query.obsm: {query_umap_key}")
    if ref_umap_key not in adata_ref.obsm:
        raise KeyError(f"Missing ref_umap_key in adata_ref.obsm: {ref_umap_key}")
    if query_label_col not in adata_query.obs:
        raise KeyError(f"Missing query_label_col in adata_query.obs: {query_label_col}")

    ref_xy = np.asarray(adata_ref.obsm[ref_umap_key], dtype=np.float32)
    qry_xy = np.asarray(adata_query.obsm[query_umap_key], dtype=np.float32)
    if ref_xy.ndim != 2 or ref_xy.shape[1] < 2:
        raise ValueError(f"ref_umap_key must be n x 2+, got shape={ref_xy.shape}")
    if qry_xy.ndim != 2 or qry_xy.shape[1] < 2:
        raise ValueError(f"query_umap_key must be n x 2+, got shape={qry_xy.shape}")
    if qry_xy.shape[0] != adata_query.n_obs:
        raise ValueError(
            "Query UMAP rows mismatch: "
            f"{qry_xy.shape[0]} (obsm) vs {adata_query.n_obs} (obs)"
        )

    labels = adata_query.obs[query_label_col].astype(str).to_numpy()
    valid_mask = np.isfinite(qry_xy[:, 0]) & np.isfinite(qry_xy[:, 1]) & (labels != "nan")
    if not np.any(valid_mask):
        raise ValueError("No valid query points after removing NaN coordinates/labels.")

    qdf = pd.DataFrame(
        {
            "umap1": qry_xy[valid_mask, 0],
            "umap2": qry_xy[valid_mask, 1],
            "label": labels[valid_mask],
        }
    )

    cmap_use: dict[str, str] = {}
    if isinstance(color_map, dict):
        cmap_use = {str(k): str(v) for k, v in color_map.items()}
    else:
        if color_uns_key is None:
            base = str(query_label_col)
            for suffix in ("_predicted", "_pred", "_pseudo"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            color_uns_key = f"{base}_colors"
        if color_uns_key in adata_ref.uns:
            base_obs = str(color_uns_key)[: -len("_colors")] if str(color_uns_key).endswith("_colors") else None
            if base_obs is not None and base_obs in adata_ref.obs:
                cats = pd.Categorical(adata_ref.obs[base_obs]).categories.tolist()
                cols = list(adata_ref.uns[color_uns_key])
                cmap_use = {str(c): str(col) for c, col in zip(cats, cols)}
        if (not cmap_use) and (color_uns_key in adata_query.uns):
            base_obs = str(color_uns_key)[: -len("_colors")] if str(color_uns_key).endswith("_colors") else None
            if base_obs is not None and base_obs in adata_query.obs:
                cats = pd.Categorical(adata_query.obs[base_obs]).categories.tolist()
                cols = list(adata_query.uns[color_uns_key])
                cmap_use = {str(c): str(col) for c, col in zip(cats, cols)}

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
    except Exception as e:
        raise RuntimeError(f"plot_projection_celltype_overlay requires matplotlib: {e}") from e

    adjust_text_fn = None
    if bool(use_adjust_text):
        try:
            from adjustText import adjust_text as adjust_text_fn  # type: ignore[assignment]
        except Exception:
            adjust_text_fn = None

    if adjust_arrowprops is None:
        adjust_arrowprops = dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "DejaVu Sans"],
        }
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
        ax.set_facecolor("white")

        ax.scatter(
            ref_xy[:, 0],
            ref_xy[:, 1],
            c=str(ref_color),
            s=float(ref_point_size),
            alpha=float(ref_alpha),
            rasterized=bool(rasterized),
            linewidths=0,
        )

        texts: list[object] = []
        centroids: dict[str, tuple[float, float]] = {}
        for cell_type, group in qdf.groupby("label", sort=False):
            clr = str(cmap_use.get(str(cell_type), default_color))
            ax.scatter(
                group["umap1"].to_numpy(),
                group["umap2"].to_numpy(),
                c=clr,
                s=float(query_point_size),
                alpha=float(query_alpha),
                edgecolors="none",
                zorder=10,
                rasterized=bool(rasterized),
            )

            cx = float(group["umap1"].mean())
            cy = float(group["umap2"].mean())
            centroids[str(cell_type)] = (cx, cy)
            txt = ax.text(
                cx,
                cy,
                str(cell_type),
                fontsize=float(label_fontsize),
                ha="center",
                va="center",
                color=clr,
                weight=str(label_weight),
                zorder=20,
            )
            txt.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=float(label_outline_width),
                        foreground="white",
                    ),
                    path_effects.Normal(),
                ]
            )
            texts.append(txt)

        if adjust_text_fn is not None and len(texts) > 0:
            adjust_text_fn(
                texts,
                arrowprops=adjust_arrowprops,
                expand_points=adjust_expand_points,
                expand_text=adjust_expand_text,
                force_points=adjust_force_points,
                force_text=adjust_force_text,
                ax=ax,
            )

        if title is not None and str(title).strip():
            ax.set_title(str(title))
        else:
            ax.set_title("")

        if remove_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)
        ax.set_aspect("equal", adjustable="box")

        fig.tight_layout()
        if save_path:
            fig.savefig(
                str(save_path),
                dpi=int(dpi),
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
        if show:
            plt.show()
        plt.close(fig)

    return {
        "query_umap_key": str(query_umap_key),
        "ref_umap_key": str(ref_umap_key),
        "query_label_col": str(query_label_col),
        "n_ref": int(ref_xy.shape[0]),
        "n_query": int(qdf.shape[0]),
        "n_labels": int(qdf["label"].nunique()),
        "color_uns_key": (None if color_uns_key is None else str(color_uns_key)),
        "save_path": (None if save_path is None else str(save_path)),
        "centroids": {k: (float(v[0]), float(v[1])) for k, v in centroids.items()},
        "adjust_text_used": bool(adjust_text_fn is not None),
    }
