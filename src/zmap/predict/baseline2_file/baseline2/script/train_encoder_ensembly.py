#!/usr/bin/env python
"""
Phase 1 encoder training + projection for ZMAP.

Trains an MLP to regress ZMAP latent (obsm) from raw/log-normalized expression
without batch correction, then projects a Sur2023 pseudo-query subset and
evaluates label agreement via kNN in latent space.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
import math
from typing import Iterable
import re

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

try:
    import umap  # type: ignore

    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _decode_arr(arr: np.ndarray) -> np.ndarray:
    out = []
    for v in arr:
        if isinstance(v, (bytes, bytearray)):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return np.array(out, dtype=object)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def read_obs_index(path: str, index_key: str = "_index") -> np.ndarray:
    with h5py.File(path, "r") as f:
        idx = f["obs"][index_key][...]
    if isinstance(idx, np.ndarray) and idx.dtype.kind in ("S", "O"):
        idx = _decode_arr(idx)
    return idx.astype(object)


def read_obs_bool(path: str, col: str) -> np.ndarray | None:
    with h5py.File(path, "r") as f:
        if col not in f["obs"]:
            return None
        g = f["obs"][col]
        if isinstance(g, h5py.Group) and "categories" in g:
            codes = g["codes"][...]
            cats = g["categories"][...]
            cats = _decode_arr(cats)
            vals = np.array([cats[i] if i >= 0 else None for i in codes])
            vals = np.array([str(v).lower() == "true" for v in vals], dtype=bool)
            return vals
        data = g[...]
        if isinstance(data, np.ndarray) and data.dtype.kind in ("S", "O"):
            data = _decode_arr(data)
            return np.array([str(v).lower() == "true" for v in data], dtype=bool)
    return data.astype(bool)


def parse_gene_list(spec: str) -> list[str]:
    if not spec:
        return []
    if os.path.exists(spec):
        genes = [g.strip() for g in open(spec).read().splitlines()]
    else:
        genes = [g.strip() for g in spec.split(",")]
    return [g for g in genes if g]


def parse_label_list(spec: str | None) -> list[str]:
    if spec is None:
        return []
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def parse_int_list(spec: str | None) -> list[int]:
    if spec is None:
        return []
    out: list[int] = []
    for tok in str(spec).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def read_id_list_csv(path: str | Path) -> list[str]:
    p = Path(path)
    with p.open(newline="") as f:
        rows = [row for row in csv.reader(f) if row and str(row[0]).strip()]
    if not rows:
        return []
    header = str(rows[0][0]).strip().lower()
    header_like = {"cell_id", "id", "obs_name", "obs_names", "index", "_index"}
    if header_like and header in header_like:
        rows = rows[1:]
    return [str(row[0]).strip() for row in rows if str(row[0]).strip()]


def hvg_genes_from_key(adata: ad.AnnData, hvg_key: str) -> list[str]:
    if hvg_key in adata.var:
        mask = adata.var[hvg_key].to_numpy()
        if mask.dtype != bool:
            mask = mask.astype(bool)
        return adata.var_names[mask].astype(str).tolist()
    genes = parse_gene_list(hvg_key)
    if not genes:
        raise ValueError(f"Missing HVG genes for key: {hvg_key}")
    var_set = {str(v) for v in adata.var_names}
    genes = [g for g in genes if g in var_set]
    if not genes:
        raise ValueError(f"No HVG genes found in var_names for key: {hvg_key}")
    return genes


def map_genes_to_indices(var_names: pd.Index, genes: list[str], label: str) -> np.ndarray:
    idx = pd.Index(var_names.astype(str)).get_indexer(genes)
    if (idx < 0).any():
        missing = [genes[i] for i in np.where(idx < 0)[0][:10]]
        raise ValueError(f"Missing {label} genes in var_names: {missing}")
    return idx.astype(int)


def _as_str(val: object) -> str:
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8")
    return str(val)


def fetch_X_batch_aligned(
    adata: ad.AnnData,
    indices: np.ndarray,
    hvg_len: int,
    layer_key: str,
    do_log1p: bool,
    do_normalize: bool,
    target_sum: float,
    scale_mean: np.ndarray | None,
    scale_std: np.ndarray | None,
    mask_hvg: np.ndarray | None,
    present_query_idx: np.ndarray,
    hvg_present_pos: np.ndarray,
    full_adata: ad.AnnData | ad.Raw | None,
    full_layer_key: str | None,
    present_full_idx: np.ndarray,
    hvg_full_pos: np.ndarray,
    missing_pos: np.ndarray,
) -> np.ndarray:
    X = np.zeros((len(indices), hvg_len), dtype=np.float32)
    scale_factors = None
    if do_normalize:
        scale_factors = compute_scale_factors(adata, indices, layer_key, target_sum)
    if present_query_idx.size > 0:
        layer = get_layer(adata, layer_key)
        X_present = layer[indices][:, present_query_idx]
        if sp.issparse(X_present):
            X_present = X_present.toarray()
        if scale_factors is not None:
            X_present = X_present * scale_factors[:, None]
        if do_log1p:
            X_present = np.log1p(X_present)
        X[:, hvg_present_pos] = np.asarray(X_present, dtype=np.float32)
    if full_adata is not None and present_full_idx.size > 0:
        if full_layer_key is None or full_layer_key == "X":
            full_X = full_adata.X
        else:
            if not hasattr(full_adata, "layers") or full_layer_key not in full_adata.layers:
                raise KeyError(f"Missing full layer: {full_layer_key}")
            full_X = full_adata.layers[full_layer_key]
        X_full = full_X[indices][:, present_full_idx]
        if sp.issparse(X_full):
            X_full = X_full.toarray()
        if scale_factors is not None:
            X_full = X_full * scale_factors[:, None]
        if do_log1p:
            X_full = np.log1p(X_full)
        X[:, hvg_full_pos] = np.asarray(X_full, dtype=np.float32)
    if scale_mean is not None and scale_std is not None:
        scale_std_safe = np.where(scale_std == 0, 1.0, scale_std)
        X = (X - scale_mean) / scale_std_safe
    if mask_hvg is not None:
        X[:, mask_hvg] = 0.0
    if missing_pos.size > 0:
        X[:, missing_pos] = 0.0
    return X


def ensure_log1p_layer(
    path: str,
    src_layer: str = "raw_nolog",
    dst_layer: str = "raw_log1p",
    chunk_size: int = 5_000_000,
) -> None:
    with h5py.File(path, "r+") as f:
        layers = f["layers"]
        if dst_layer in layers:
            return
        if src_layer not in layers:
            raise KeyError(f"Missing source layer: {src_layer}")

        src = layers[src_layer]
        dst = layers.create_group(dst_layer)
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        f.copy(src["indices"], dst, name="indices")
        f.copy(src["indptr"], dst, name="indptr")

        data = src["data"]
        dtype = data.dtype
        if not np.issubdtype(dtype, np.floating):
            dtype = np.float32
        dset = dst.create_dataset(
            "data",
            shape=data.shape,
            dtype=dtype,
            chunks=True,
            compression=data.compression,
            compression_opts=data.compression_opts,
        )
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i : i + chunk_size]
            if not np.issubdtype(chunk.dtype, np.floating):
                chunk = chunk.astype(np.float32)
            chunk = np.log1p(chunk).astype(dtype, copy=False)
            dset[i : i + chunk_size] = chunk


def load_hvg_indices(adata: ad.AnnData, hvg_key: str) -> np.ndarray:
    if hvg_key not in adata.var:
        raise KeyError(f"Missing var key for HVGs: {hvg_key}")
    mask = adata.var[hvg_key].to_numpy()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return np.where(mask)[0]


def get_layer(adata: ad.AnnData, layer_key: str) -> sp.spmatrix | np.ndarray:
    if layer_key == "X":
        return adata.X
    if layer_key not in adata.layers:
        raise KeyError(f"Missing layer: {layer_key}")
    return adata.layers[layer_key]


def compute_scale_factors(
    adata: ad.AnnData, indices: np.ndarray, layer_key: str, target_sum: float
) -> np.ndarray:
    layer = get_layer(adata, layer_key)
    X = layer[indices]
    if sp.issparse(X):
        sums = np.asarray(X.sum(axis=1)).ravel()
    else:
        sums = np.asarray(np.sum(X, axis=1)).ravel()
    scales = np.zeros_like(sums, dtype=np.float32)
    nonzero = sums > 0
    scales[nonzero] = target_sum / sums[nonzero]
    return scales


def fetch_X_batch(
    adata: ad.AnnData,
    indices: np.ndarray,
    hvg_idx: np.ndarray,
    layer_key: str,
    do_log1p: bool,
    do_normalize: bool,
    target_sum: float,
    scale_mean: np.ndarray | None,
    scale_std: np.ndarray | None,
    mask_hvg: np.ndarray | None,
    hvg_cache: np.ndarray | None = None,
    libsize_cache: np.ndarray | None = None,
) -> np.ndarray:
    if hvg_cache is not None:
        X = hvg_cache[indices]
    else:
        layer = get_layer(adata, layer_key)
        X = layer[indices][:, hvg_idx]
        if sp.issparse(X):
            X = X.toarray()
    if do_normalize:
        if libsize_cache is not None:
            sums = libsize_cache[indices]
            scale_factors = np.zeros_like(sums, dtype=np.float32)
            nonzero = sums > 0
            scale_factors[nonzero] = target_sum / sums[nonzero]
        else:
            scale_factors = compute_scale_factors(adata, indices, layer_key, target_sum)
        X = X * scale_factors[:, None]
    if do_log1p:
        X = np.log1p(X)
    if scale_mean is not None and scale_std is not None:
        scale_std_safe = np.where(scale_std == 0, 1.0, scale_std)
        X = (X - scale_mean) / scale_std_safe
    if mask_hvg is not None:
        X[:, mask_hvg] = 0.0
    return np.asarray(X, dtype=np.float32)


def apply_input_scale_and_mask(
    X: np.ndarray,
    scale_mean: np.ndarray | None,
    scale_std: np.ndarray | None,
    mask_hvg: np.ndarray | None,
) -> np.ndarray:
    Y = np.asarray(X, dtype=np.float32).copy()
    if scale_mean is not None and scale_std is not None:
        scale_std_safe = np.where(scale_std == 0, 1.0, scale_std)
        Y = (Y - scale_mean) / scale_std_safe
    if mask_hvg is not None:
        Y[:, mask_hvg] = 0.0
    return np.asarray(Y, dtype=np.float32)


def fetch_Z_batch(adata: ad.AnnData, indices: np.ndarray, latent_key: str) -> np.ndarray:
    Z = adata.obsm[latent_key][indices]
    return np.asarray(Z, dtype=np.float32)


def iter_batches(
    indices: np.ndarray, batch_size: int, rng: np.random.Generator, shuffle: bool
) -> Iterable[np.ndarray]:
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


class MLP(torch.nn.Module):
    class ResidualBlock(torch.nn.Module):
        def __init__(
            self,
            dim: int,
            hidden_layernorm: bool = True,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(dim, dim)
            self.ln1 = torch.nn.LayerNorm(dim) if hidden_layernorm else None
            self.act = torch.nn.GELU()
            self.drop = torch.nn.Dropout(dropout) if dropout and dropout > 0 else None
            self.fc2 = torch.nn.Linear(dim, dim)
            self.ln2 = torch.nn.LayerNorm(dim) if hidden_layernorm else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.fc1(x)
            if self.ln1 is not None:
                y = self.ln1(y)
            y = self.act(y)
            if self.drop is not None:
                y = self.drop(y)
            y = self.fc2(y)
            if self.ln2 is not None:
                y = self.ln2(y)
            return x + y

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        out_dim: int,
        use_layernorm: bool = False,
        hidden_layernorm: bool = True,
        dropout: float = 0.0,
        residual_blocks: int = 0,
    ) -> None:
        super().__init__()
        # Input LayerNorm is optional; hidden LayerNorm can be toggled.
        self.ln_in = torch.nn.LayerNorm(in_dim) if use_layernorm else None
        hidden_layers: list[torch.nn.Module] = []
        prev_dim = in_dim
        for h in hidden:
            hidden_layers.append(torch.nn.Linear(prev_dim, h))
            if hidden_layernorm:
                hidden_layers.append(torch.nn.LayerNorm(h))
            hidden_layers.append(torch.nn.GELU())
            if dropout and dropout > 0:
                hidden_layers.append(torch.nn.Dropout(dropout))
            prev_dim = h
        self.hidden = (
            torch.nn.Sequential(*hidden_layers) if hidden_layers else torch.nn.Identity()
        )

        n_res = max(0, int(residual_blocks))
        if n_res > 0:
            blocks: list[torch.nn.Module] = []
            for _ in range(n_res):
                blocks.append(
                    MLP.ResidualBlock(
                        prev_dim,
                        hidden_layernorm=hidden_layernorm,
                        dropout=dropout,
                    )
                )
            self.res_blocks = torch.nn.Sequential(*blocks)
        else:
            self.res_blocks = torch.nn.Identity()

        self.head = torch.nn.Linear(prev_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ln_in is not None:
            x = self.ln_in(x)
        x = self.hidden(x)
        x = self.res_blocks(x)
        return self.head(x)


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * float(ctx.coeff), None


def grad_reverse(x: torch.Tensor, coeff: float) -> torch.Tensor:
    return GradientReversalFn.apply(x, float(coeff))


class DomainClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = int(in_dim)
        for h in hidden:
            hh = int(h)
            if hh <= 0:
                continue
            layers.append(torch.nn.Linear(prev, hh))
            layers.append(torch.nn.LayerNorm(hh))
            layers.append(torch.nn.GELU())
            if float(dropout) > 0:
                layers.append(torch.nn.Dropout(float(dropout)))
            prev = hh
        self.hidden = torch.nn.Sequential(*layers) if layers else torch.nn.Identity()
        self.head = torch.nn.Linear(prev, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))


def get_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    ln_param_ids: set[int] = set()
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters(recurse=False):
                ln_param_ids.add(id(p))
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or id(param) in ln_param_ids:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    decay_ids = {id(p) for p in decay_params}
    no_decay_ids = {id(p) for p in no_decay_params}
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert decay_ids.isdisjoint(no_decay_ids)
    assert decay_ids | no_decay_ids == all_ids
    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_module_target_matrix(
    module_csv: str,
    hvg_genes: list[str],
    min_genes_per_module: int = 3,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    df = pd.read_csv(module_csv)
    required = {"gene", "celltype"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"module marker csv missing columns: {missing}")
    if "module_level" not in df.columns:
        df["module_level"] = "module"
    if "module_weight" not in df.columns:
        sr = df["support_ratio"].astype(float).clip(lower=0.0) if "support_ratio" in df.columns else 1.0
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

    gene_to_idx = {g: i for i, g in enumerate(hvg_genes)}
    n_hvg = len(hvg_genes)

    # Aggregate per module + gene and normalize inside each module.
    x = (
        df.groupby(["module_level", "celltype", "gene"], as_index=False)["module_weight"]
        .sum()
        .sort_values(["module_level", "celltype", "module_weight"], ascending=[True, True, False])
    )
    x["hvg_idx"] = x["gene"].map(gene_to_idx).astype("Int64")

    module_names: list[str] = []
    module_levels: list[str] = []
    module_gene_counts: list[int] = []
    cols: list[np.ndarray] = []
    for (lvl, ct), g in x.groupby(["module_level", "celltype"], sort=False):
        gg = g[g["hvg_idx"].notna()].copy()
        if gg.empty:
            continue
        idx = gg["hvg_idx"].astype(int).to_numpy()
        w = gg["module_weight"].to_numpy(dtype=np.float32)
        # Re-normalize after HVG intersection.
        denom = float(np.sum(w))
        if denom <= 0:
            continue
        w = w / denom
        if idx.size < int(min_genes_per_module):
            continue
        col = np.zeros(n_hvg, dtype=np.float32)
        col[idx] = w
        cols.append(col)
        module_names.append(f"{lvl}|{ct}")
        module_levels.append(str(lvl))
        module_gene_counts.append(int(idx.size))

    if not cols:
        raise ValueError(
            "No valid modules after HVG intersection and min-gene filter. "
            "Check module CSV and --module-min-genes."
        )

    W = np.stack(cols, axis=1).astype(np.float32)  # [n_hvg, n_modules]
    return W, module_names, module_levels, module_gene_counts


def label_transfer_knn(
    ref_latent: np.ndarray,
    ref_labels: np.ndarray,
    query_latent: np.ndarray,
    k: int,
) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(ref_latent, ref_labels)
    return knn


def tissue_conditioned_knn_predict(
    ref_latent: np.ndarray,
    ref_labels: np.ndarray,
    ref_tissue: np.ndarray | None,
    query_latent: np.ndarray,
    query_tissue: np.ndarray | None,
    k: int,
    mode: str,
    penalty_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx, dist = tissue_conditioned_knn_neighbors(
        ref_latent,
        ref_tissue,
        query_latent,
        query_tissue,
        k,
        mode,
        penalty_lambda,
    )
    return knn_labels_from_neighbors(ref_labels, idx, dist)


def knn_predict_with_confidence(
    knn: KNeighborsClassifier, query_latent: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = knn.predict(query_latent)
    proba = knn.predict_proba(query_latent)
    max_prob = proba.max(axis=1)
    entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
    dist, _ = knn.kneighbors(query_latent, n_neighbors=knn.n_neighbors)
    mean_dist = dist.mean(axis=1)
    return pred, max_prob, entropy, mean_dist


def knn_labels_from_neighbors(
    ref_labels: np.ndarray, idx: np.ndarray, dist: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_query = idx.shape[0]
    pred = np.full(n_query, "unassigned", dtype=object)
    max_prob = np.zeros(n_query, dtype=np.float32)
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
        total = weights.sum()
        if total <= 0:
            continue
        label_weights: dict[str, float] = {}
        for lbl, w in zip(ref_labels[row_idx], weights):
            label_weights[lbl] = label_weights.get(lbl, 0.0) + float(w)
        best_label = max(label_weights.items(), key=lambda x: x[1])[0]
        probs = np.array(list(label_weights.values()), dtype=np.float64) / total
        pred[i] = best_label
        max_prob[i] = float(probs.max())
        entropy[i] = float(-np.sum(probs * np.log(probs + 1e-12)))
        mean_dist[i] = float(row_dist.mean())
    return pred, max_prob, entropy, mean_dist


def tissue_conditioned_knn_neighbors(
    ref_latent: np.ndarray,
    ref_tissue: np.ndarray | None,
    query_latent: np.ndarray,
    query_tissue: np.ndarray | None,
    k: int,
    mode: str,
    penalty_lambda: float,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    n_query = query_latent.shape[0]
    idx_out = np.full((n_query, k), -1, dtype=np.int64)
    dist_out = np.full((n_query, k), np.nan, dtype=np.float32)

    if ref_latent.shape[0] == 0:
        return idx_out, dist_out

    if ref_tissue is None or query_tissue is None or mode == "none":
        nn = NearestNeighbors(n_neighbors=min(k, ref_latent.shape[0]), metric=metric)
        nn.fit(ref_latent)
        dist, idx = nn.kneighbors(query_latent, return_distance=True)
        k_use = dist.shape[1]
        idx_out[:, :k_use] = idx
        dist_out[:, :k_use] = dist
        return idx_out, dist_out

    ref_tissue = ref_tissue.astype(str)
    query_tissue = query_tissue.astype(str)
    unique_tissues = pd.unique(query_tissue)

    if mode == "soft":
        global_nn = NearestNeighbors(n_neighbors=min(k, ref_latent.shape[0]), metric=metric)
        global_nn.fit(ref_latent)
        tissue_models: dict[str, tuple[NearestNeighbors, np.ndarray]] = {}
        for tissue in unique_tissues:
            r_mask = ref_tissue == tissue
            r_idx = np.where(r_mask)[0]
            if r_idx.size == 0:
                continue
            nn = NearestNeighbors(n_neighbors=min(k, r_idx.size), metric=metric)
            nn.fit(ref_latent[r_idx])
            tissue_models[tissue] = (nn, r_idx)

        for tissue in unique_tissues:
            q_mask = query_tissue == tissue
            if not np.any(q_mask):
                continue
            q_lat = query_latent[q_mask]
            g_dist, g_idx = global_nn.kneighbors(q_lat, return_distance=True)
            s_idx = None
            s_dist = None
            if tissue in tissue_models:
                nn, r_idx = tissue_models[tissue]
                s_dist, s_loc = nn.kneighbors(q_lat, return_distance=True)
                s_idx = r_idx[s_loc]

            q_rows = np.flatnonzero(q_mask)
            for row in range(q_lat.shape[0]):
                cand: dict[int, float] = {}
                for j, idx in enumerate(g_idx[row]):
                    d = float(g_dist[row, j])
                    cand[idx] = d if idx not in cand else min(cand[idx], d)
                if s_idx is not None and s_dist is not None:
                    for j, idx in enumerate(s_idx[row]):
                        d = float(s_dist[row, j])
                        cand[idx] = d if idx not in cand else min(cand[idx], d)

                if not cand:
                    continue
                cand_idx = np.fromiter(cand.keys(), dtype=np.int64)
                cand_dist = np.fromiter(cand.values(), dtype=np.float32)
                same = ref_tissue[cand_idx] == tissue
                penalty = penalty_lambda * (1.0 - same.astype(np.float32))
                d_soft = cand_dist + penalty

                order = np.argsort(d_soft)[:k]
                sel_idx = cand_idx[order]
                sel_dist = d_soft[order]
                row_idx = q_rows[row]
                idx_out[row_idx, : sel_idx.size] = sel_idx
                dist_out[row_idx, : sel_dist.size] = sel_dist
        return idx_out, dist_out

    global_nn = NearestNeighbors(n_neighbors=min(k, ref_latent.shape[0]))
    global_nn.fit(ref_latent)
    for tissue in unique_tissues:
        q_mask = query_tissue == tissue
        if not np.any(q_mask):
            continue
        r_mask = ref_tissue == tissue
        ref_count = int(r_mask.sum())
        if ref_count >= max(1, k):
            nn = NearestNeighbors(n_neighbors=min(k, ref_count))
            nn.fit(ref_latent[r_mask])
            dist, loc = nn.kneighbors(query_latent[q_mask], return_distance=True)
            idx = np.where(r_mask)[0][loc]
        else:
            dist, idx = global_nn.kneighbors(query_latent[q_mask], return_distance=True)
        k_use = dist.shape[1]
        q_rows = np.flatnonzero(q_mask)
        idx_out[q_rows, :k_use] = idx
        dist_out[q_rows, :k_use] = dist
    return idx_out, dist_out


def latent_label_distance_mask(
    ref_latent: np.ndarray,
    ref_labels: np.ndarray | None,
    query_latent: np.ndarray,
    label: str | None,
    metric: str,
    quantile: float,
    k: int,
) -> np.ndarray | None:
    if not (0 < quantile < 1):
        raise ValueError("--latent-filter-quantile must be between 0 and 1.")
    if ref_latent.shape[0] == 0:
        print("Latent filter skipped: empty reference latent.")
        return None
    label_norm = label.strip().lower() if label else ""
    if label_norm:
        labels = np.asarray(ref_labels, dtype=str)
        mask = np.char.lower(labels) == label_norm
        if not np.any(mask):
            print(f"Latent filter skipped: label '{label}' not found in reference.")
            return None
        ref_subset = ref_latent[mask]
        label_desc = label_norm
    else:
        ref_subset = ref_latent
        label_desc = "all_ref"
    if ref_subset.shape[0] == 0:
        print(f"Latent filter skipped: no reference cells for '{label_desc}'.")
        return None
    k_use = min(max(1, k), ref_subset.shape[0])
    nn = NearestNeighbors(n_neighbors=k_use, metric=metric)
    nn.fit(ref_subset)
    dist, _ = nn.kneighbors(query_latent, return_distance=True)
    mean_dist = dist.mean(axis=1)
    thr = float(np.quantile(mean_dist, quantile))
    keep = mean_dist <= thr
    kept = int(np.sum(keep))
    print(
        f"Latent filter ({label_desc}): metric={metric} k={k_use} "
        f"quantile={quantile} thr={thr:.4f} kept={kept}/{keep.size}"
    )
    return keep


def _softmax_weights(dist: np.ndarray, tau: float) -> np.ndarray:
    if dist.size == 0:
        return dist
    if tau <= 0:
        tau_vec = np.median(dist, axis=1, keepdims=True)
        tau_vec = np.maximum(tau_vec, 1e-8)
    else:
        tau_vec = np.full((dist.shape[0], 1), max(tau, 1e-8), dtype=np.float32)
    weights = np.exp(-dist / tau_vec)
    denom = weights.sum(axis=1, keepdims=True)
    weights = np.where(denom > 0, weights / denom, 0.0)
    return weights


def _infer_time_order(categories: list[str]) -> list[str]:
    def extract_num(label: str) -> float | None:
        match = re.search(r"[-+]?\d*\.?\d+", label)
        return float(match.group()) if match else None

    nums = [extract_num(c) for c in categories]
    if all(n is not None for n in nums):
        return [c for _, c in sorted(zip(nums, categories), key=lambda x: (x[0], x[1]))]
    return sorted(categories)


def time_regression_from_neighbors(
    ref_time: np.ndarray,
    idx: np.ndarray,
    dist: np.ndarray,
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
        tau_i = np.median(row_dist) if tau <= 0 else max(tau, 1e-8)
        weights = np.exp(-row_dist / max(tau_i, 1e-8))
        denom = weights.sum()
        if denom <= 0:
            continue
        weights = weights / denom
        t_vals = ref_time[row_idx]
        valid_time = np.isfinite(t_vals)
        if not np.any(valid_time):
            continue
        weights = weights * valid_time
        denom = weights.sum()
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
                weights[mask] *= monotone_gamma
                denom = weights.sum()
                if denom <= 0:
                    continue
                weights = weights / denom
        if topk > 0:
            keep = min(topk, weights.size)
            top_idx = np.argsort(weights)[-keep:]
            mask = np.zeros_like(weights, dtype=bool)
            mask[top_idx] = True
            weights = np.where(mask, weights, 0.0)
            denom = weights.sum()
            if denom <= 0:
                continue
            weights = weights / denom
        if trim_extremes > 0:
            keep = weights > 0
            if np.sum(keep) > trim_extremes * 2:
                t_vals_trim = t_vals[keep]
                w_trim = weights[keep]
                order_idx = np.argsort(t_vals_trim)
                trim = min(trim_extremes, (len(order_idx) - 1) // 2)
                if trim > 0:
                    drop_idx = list(order_idx[:trim]) + list(order_idx[-trim:])
                    keep_mask = np.ones_like(w_trim, dtype=bool)
                    keep_mask[drop_idx] = False
                    w_trim = w_trim * keep_mask
                    denom = w_trim.sum()
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
        mean_dist[i] = float(row_dist.mean())
        for t, w in zip(t_vals, w_vals):
            t_int = int(t)
            if 0 <= t_int < n_classes:
                time_prob[i, t_int] += float(w)
    return time_pred, time_var, mean_dist, time_prob


def time_regression_knn(
    ref_latent: np.ndarray,
    ref_time: np.ndarray,
    ref_tissue: np.ndarray | None,
    query_latent: np.ndarray,
    query_tissue: np.ndarray | None,
    k: int,
    mode: str,
    penalty_lambda: float,
    tau: float,
    topk: int,
    query_time_enc: np.ndarray | None,
    monotone_delta: int,
    monotone_gamma: float,
    trim_extremes: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx, dist = tissue_conditioned_knn_neighbors(
        ref_latent,
        ref_tissue,
        query_latent,
        query_tissue,
        k,
        mode,
        penalty_lambda,
        metric,
    )
    n_classes = int(np.nanmax(ref_time) + 1) if ref_time.size else 0
    time_pred, time_var, mean_dist, _ = time_regression_from_neighbors(
        ref_time,
        idx,
        dist,
        tau,
        n_classes,
        topk,
        query_time_enc,
        monotone_delta,
        monotone_gamma,
        trim_extremes,
    )
    return time_pred, time_var, mean_dist


def save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_dir: str,
    prefix: str = "",
    omit_true_labels: list[str] | None = None,
) -> None:
    ensure_dir(out_dir)
    if omit_true_labels:
        omit_norm = {str(x).strip().lower() for x in omit_true_labels if str(x).strip()}
        if omit_norm:
            keep = np.array(
                [str(v).strip().lower() not in omit_norm for v in y_true], dtype=bool
            )
            y_true = y_true[keep]
            y_pred = y_pred[keep]
            labels = [
                lbl for lbl in labels if str(lbl).strip().lower() not in omit_norm
            ]
    suffix = f"_{prefix}" if prefix else ""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(out_dir, f"confusion_matrix{suffix}.csv"))
    plot_confusion_matrix(cm, labels, out_dir, suffix)

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(os.path.join(out_dir, f"classification_report{suffix}.csv"))
    plot_classification_report(report_df, labels, out_dir, suffix)

    acc = accuracy_score(y_true, y_pred)
    with open(os.path.join(out_dir, f"overall_accuracy{suffix}.txt"), "w") as f:
        f.write(f"{acc:.6f}\n")


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    out_dir: str,
    suffix: str,
    normalize: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_size = max(8.0, min(30.0, len(labels) * 0.35))
    if normalize:
        col_sums = cm.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = np.where(col_sums > 0, (cm / col_sums), 0.0)
    else:
        cm_plot = cm.astype(float)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_plot, aspect="auto", cmap="Blues")
    if cm_plot.size:
        row_max_idx = np.argmax(cm_plot, axis=1)
        for i, j in enumerate(row_max_idx):
            if normalize and col_sums[0, j] == 0:
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
    if len(labels) <= 120:
        ticks = np.arange(len(labels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(x) for x in labels], fontsize=6, rotation=90)
        ax.set_yticklabels([str(x) for x in labels], fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"confusion_matrix{suffix}.png"), dpi=200)
    plt.close(fig)


def plot_classification_report(
    report_df: pd.DataFrame, labels: list[str], out_dir: str, suffix: str
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    keep_labels = [lbl for lbl in labels if lbl in report_df.index]
    if not keep_labels:
        return

    metrics = ["precision", "recall", "f1-score"]
    plot_df = report_df.loc[keep_labels, metrics].fillna(0.0)
    order = plot_df.sort_values("f1-score", ascending=False).index.tolist()
    plot_df = plot_df.loc[order]
    fig_w = max(10.0, min(40.0, len(order) * 0.25))
    fig_h = 9.0
    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=1, figsize=(fig_w, fig_h), sharex=True
    )
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(order))
    for ax, metric in zip(axes, metrics):
        ax.bar(x, plot_df[metric].to_numpy(), color="#4C78A8")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric.capitalize())
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(order, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"classification_report_plot{suffix}.png"), dpi=300
    )
    plt.close(fig)


def compute_latent_stats(
    adata: ad.AnnData,
    indices: np.ndarray,
    latent_key: str,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    mean = None
    m2 = None
    for batch_idx in iter_batches(indices, batch_size, rng, False):
        Z = fetch_Z_batch(adata, batch_idx, latent_key)
        if mean is None:
            mean = np.zeros(Z.shape[1], dtype=np.float64)
            m2 = np.zeros(Z.shape[1], dtype=np.float64)
        count_new = count + Z.shape[0]
        delta = Z - mean
        mean += delta.sum(axis=0) / count_new
        delta2 = Z - mean
        m2 += (delta * delta2).sum(axis=0)
        count = count_new
    if mean is None or m2 is None:
        raise ValueError("No samples to compute latent stats.")
    var = m2 / max(1, count - 1)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def compute_library_size(
    adata: ad.AnnData,
    indices: np.ndarray,
    layer_key: str,
    batch_size: int,
) -> np.ndarray:
    libsize = np.zeros(adata.n_obs, dtype=np.float32)
    layer = get_layer(adata, layer_key)
    for batch_idx in iter_batches(indices, batch_size, np.random.default_rng(0), False):
        X = layer[batch_idx]
        if sp.issparse(X):
            sums = np.asarray(X.sum(axis=1)).ravel()
        else:
            sums = np.asarray(np.sum(X, axis=1)).ravel()
        libsize[batch_idx] = sums.astype(np.float32)
    return libsize


def build_hvg_cache(
    adata: ad.AnnData,
    indices: np.ndarray,
    hvg_idx: np.ndarray,
    layer_key: str,
    batch_size: int,
) -> np.ndarray:
    n = adata.n_obs
    hvg_cache = np.zeros((n, len(hvg_idx)), dtype=np.float32)
    layer = get_layer(adata, layer_key)
    for batch_idx in iter_batches(indices, batch_size, np.random.default_rng(0), False):
        X = layer[batch_idx][:, hvg_idx]
        if sp.issparse(X):
            X = X.toarray()
        hvg_cache[batch_idx] = np.asarray(X, dtype=np.float32)
    return hvg_cache

def compute_scale_stats(
    adata: ad.AnnData,
    indices: np.ndarray,
    hvg_idx: np.ndarray,
    layer_key: str,
    do_log1p: bool,
    do_normalize: bool,
    target_sum: float,
    batch_size: int,
    rng: np.random.Generator,
    hvg_cache: np.ndarray | None = None,
    libsize_cache: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    sum_vec = None
    sumsq_vec = None
    for batch_idx in iter_batches(indices, batch_size, rng, False):
        X = fetch_X_batch(
            adata,
            batch_idx,
            hvg_idx,
            layer_key,
            do_log1p,
            do_normalize,
            target_sum,
            None,
            None,
            None,
            hvg_cache=hvg_cache,
            libsize_cache=libsize_cache,
        )
        if sum_vec is None:
            sum_vec = np.zeros(X.shape[1], dtype=np.float64)
            sumsq_vec = np.zeros(X.shape[1], dtype=np.float64)
        sum_vec += X.sum(axis=0)
        sumsq_vec += (X * X).sum(axis=0)
        count += X.shape[0]
    if count == 0 or sum_vec is None or sumsq_vec is None:
        raise ValueError("No samples to compute scale stats.")
    mean = sum_vec / count
    var = sumsq_vec / count - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def coarse_contrastive_loss(
    z: torch.Tensor,
    labels: np.ndarray,
    margin: float,
    max_samples: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    valid_mask = np.array(
        [not pd.isna(v) and str(v) != "unassigned" for v in labels], dtype=bool
    )
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return torch.tensor(0.0, device=z.device)
    if len(valid_idx) > max_samples:
        valid_idx = rng.choice(valid_idx, size=max_samples, replace=False)
    z_sub = z[torch.as_tensor(valid_idx, device=z.device)]
    labels_sub = labels[valid_idx].astype(str)
    dist = torch.cdist(z_sub, z_sub)
    uniq = {lbl: i for i, lbl in enumerate(np.unique(labels_sub))}
    label_ids = np.array([uniq[lbl] for lbl in labels_sub], dtype=np.int64)
    label_col = torch.as_tensor(label_ids, device=z.device)
    same = label_col[:, None] == label_col[None, :]
    diag = torch.eye(dist.shape[0], device=z.device, dtype=torch.bool)
    pos = dist[same & ~diag]
    neg = dist[~same]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, device=z.device)
    pos_loss = pos.mean()
    neg_loss = torch.relu(margin - neg).mean()
    return pos_loss + neg_loss


def mse_within_labels(
    pred: torch.Tensor, target: torch.Tensor, labels: np.ndarray
) -> torch.Tensor:
    valid_mask = np.array(
        [not pd.isna(v) and str(v) != "unassigned" for v in labels], dtype=bool
    )
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return torch.tensor(0.0, device=pred.device)
    labels_sub = labels[valid_idx].astype(str)
    uniq = np.unique(labels_sub)
    losses = []
    for lbl in uniq:
        idx = valid_idx[labels_sub == lbl]
        if idx.size == 0:
            continue
        idx_t = torch.as_tensor(idx, device=pred.device)
        diff = pred[idx_t] - target[idx_t]
        losses.append((diff * diff).mean())
    if not losses:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).mean()


def coral_alignment_loss(
    src: torch.Tensor,
    tgt: torch.Tensor,
    mean_weight: float = 1.0,
) -> torch.Tensor:
    if src.numel() == 0 or tgt.numel() == 0:
        return torch.tensor(0.0, device=src.device if src.numel() else tgt.device)
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(
            f"CORAL feature dim mismatch: src={src.shape[1]} vs tgt={tgt.shape[1]}"
        )

    src_mean = src.mean(dim=0, keepdim=True)
    tgt_mean = tgt.mean(dim=0, keepdim=True)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    n_src = max(1, src.shape[0] - 1)
    n_tgt = max(1, tgt.shape[0] - 1)
    cov_src = (src_c.T @ src_c) / float(n_src)
    cov_tgt = (tgt_c.T @ tgt_c) / float(n_tgt)

    cov_loss = ((cov_src - cov_tgt) ** 2).mean()
    if float(mean_weight) <= 0:
        return cov_loss
    mean_loss = ((src_mean - tgt_mean) ** 2).mean()
    return cov_loss + float(mean_weight) * mean_loss


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xx = (x * x).sum(dim=1, keepdim=True)
    yy = (y * y).sum(dim=1, keepdim=True).T
    d = xx + yy - 2.0 * (x @ y.T)
    return torch.clamp(d, min=0.0)


def _infer_rbf_gamma(src: torch.Tensor, tgt: torch.Tensor) -> float:
    z = torch.cat([src, tgt], dim=0)
    if z.shape[0] < 2:
        return 1.0
    d = _pairwise_sq_dists(z, z)
    tri_mask = torch.triu(
        torch.ones(d.shape[0], d.shape[0], device=d.device, dtype=torch.bool), diagonal=1
    )
    vals = d[tri_mask]
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return 1.0
    med = float(torch.median(vals).item())
    if med <= 0:
        return 1.0
    return 1.0 / med


def mmd_alignment_loss(
    src: torch.Tensor,
    tgt: torch.Tensor,
    kernel: str = "rbf",
    gamma: float = 0.0,
) -> torch.Tensor:
    if src.numel() == 0 or tgt.numel() == 0:
        return torch.tensor(0.0, device=src.device if src.numel() else tgt.device)
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(
            f"MMD feature dim mismatch: src={src.shape[1]} vs tgt={tgt.shape[1]}"
        )
    if kernel == "linear":
        return ((src.mean(dim=0) - tgt.mean(dim=0)) ** 2).mean()

    gamma_eff = float(gamma) if float(gamma) > 0 else _infer_rbf_gamma(src, tgt)
    d_xx = _pairwise_sq_dists(src, src)
    d_yy = _pairwise_sq_dists(tgt, tgt)
    d_xy = _pairwise_sq_dists(src, tgt)
    k_xx = torch.exp(-gamma_eff * d_xx)
    k_yy = torch.exp(-gamma_eff * d_yy)
    k_xy = torch.exp(-gamma_eff * d_xy)

    m = int(src.shape[0])
    n = int(tgt.shape[0])
    if m > 1:
        sum_xx = k_xx.sum() - torch.diagonal(k_xx).sum()
        term_xx = sum_xx / float(m * (m - 1))
    else:
        term_xx = torch.tensor(0.0, device=src.device)
    if n > 1:
        sum_yy = k_yy.sum() - torch.diagonal(k_yy).sum()
        term_yy = sum_yy / float(n * (n - 1))
    else:
        term_yy = torch.tensor(0.0, device=src.device)
    term_xy = k_xy.mean()
    return term_xx + term_yy - 2.0 * term_xy


def evaluate_label_level(
    level: str,
    ref_latent: np.ndarray,
    ref_labels: np.ndarray,
    query_latent: np.ndarray,
    query_labels: np.ndarray,
    k: int,
    out_dir: str,
    exclude_unassigned: bool,
    confidence_threshold: float,
    ref_tissue: np.ndarray | None = None,
    query_tissue: np.ndarray | None = None,
    tissue_mode: str = "none",
    penalty_lambda: float = 1.0,
    omit_true_labels: list[str] | None = None,
) -> float:
    pred, max_prob, _, _ = tissue_conditioned_knn_predict(
        ref_latent,
        ref_labels,
        ref_tissue,
        query_latent,
        query_tissue,
        k,
        tissue_mode,
        penalty_lambda,
    )
    if confidence_threshold > 0:
        pred = pred.astype(object)
        pred[max_prob < confidence_threshold] = "unassigned"
    if exclude_unassigned:
        mask = (query_labels != "unassigned") & ~pd.isna(query_labels)
        query_labels = query_labels[mask]
        pred = pred[mask]
    # Report labels based on query ground truth classes (not full ref class universe).
    label_list = sorted(np.unique(query_labels).tolist())
    if exclude_unassigned:
        label_list = [lbl for lbl in label_list if lbl != "unassigned"]
    save_metrics(
        query_labels,
        pred,
        label_list,
        out_dir,
        prefix=level,
        omit_true_labels=omit_true_labels,
    )
    if omit_true_labels:
        omit_norm = {str(x).strip().lower() for x in omit_true_labels if str(x).strip()}
        keep = np.array(
            [str(v).strip().lower() not in omit_norm for v in query_labels], dtype=bool
        )
        query_labels = query_labels[keep]
        pred = pred[keep]
    return accuracy_score(query_labels, pred)


def project_query_umap_knn(
    ref_latent: np.ndarray,
    ref_umap: np.ndarray,
    query_latent: np.ndarray,
    k: int,
    weight_mode: str,
    tau: float,
    tau_quantile: float,
    ref_adata: ad.AnnData | None = None,
    proj_mode: str = "medoid",
    medoid_topm: int = 1,
    medoid_jitter: float = 0.0,
    medoid_jitter_seed: int | None = None,
    knn_idx: np.ndarray | None = None,
    knn_dist: np.ndarray | None = None,
    label_filter: bool = False,
    ref_labels: np.ndarray | None = None,
    query_pred_labels: np.ndarray | None = None,
) -> np.ndarray:
    if ref_adata is not None and ref_latent.shape[0] != ref_adata.n_obs:
        raise ValueError("Reference latent/adata size mismatch.")
    if ref_adata is None and ref_latent.shape[0] != ref_umap.shape[0]:
        raise ValueError("Reference latent/UMAP size mismatch.")
    if ref_latent.shape[0] == 0:
        raise ValueError("Empty reference latent.")
    if knn_idx is not None and knn_dist is not None:
        idx = knn_idx
        dist = knn_dist
        if idx.shape != dist.shape:
            raise ValueError("kNN idx/dist shape mismatch.")
        n_query = idx.shape[0]
        k = idx.shape[1]
        if query_latent.shape[0] != n_query:
            raise ValueError("kNN rows do not match query_latent size.")
    else:
        k = min(max(1, k), ref_latent.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(ref_latent)
        dist, idx = nn.kneighbors(query_latent, return_distance=True)
        n_query = idx.shape[0]
    nbr_pos = None
    if ref_adata is not None:
        if "X_umap" not in ref_adata.obsm:
            raise KeyError("Reference adata missing X_umap.")
        assert ref_adata.obs_names.is_unique
        ref_ids = ref_adata.obs_names.astype(str).to_numpy()
        nbr_ids = ref_ids[idx]
        if nbr_ids.shape[0] > 0:
            rng = np.random.default_rng(0)
            check_rows = rng.choice(nbr_ids.shape[0], size=min(3, nbr_ids.shape[0]), replace=False)
            for r in check_rows:
                assert (ref_ids[idx[r]] == nbr_ids[r]).all()
        ref_pos = {cid: i for i, cid in enumerate(ref_ids)}
        nbr_pos = np.array([[ref_pos.get(cid, -1) for cid in row] for row in nbr_ids], dtype=int)
        assert np.all(nbr_pos >= 0)

    if label_filter:
        if ref_labels is None or query_pred_labels is None:
            raise ValueError("Label filter requires ref_labels and query_pred_labels.")
        if ref_labels.shape[0] != ref_latent.shape[0]:
            raise ValueError("ref_labels size mismatch with ref_latent.")
        query_pred_labels = np.asarray(query_pred_labels)
        if query_pred_labels.shape[0] != n_query:
            raise ValueError("query_pred_labels size mismatch with query_latent.")
        coords = np.full((n_query, 2), np.nan, dtype=np.float32)
        rng_jitter = None
        if medoid_jitter > 0:
            rng_jitter = np.random.default_rng(medoid_jitter_seed)
        for i in range(n_query):
            row_idx = idx[i]
            row_dist = dist[i]
            valid = row_idx >= 0
            if not np.any(valid):
                continue
            row_idx = row_idx[valid]
            row_dist = row_dist[valid]
            row_pos = None
            if nbr_pos is not None:
                row_pos = nbr_pos[i][valid]
            use_idx = row_idx
            use_dist = row_dist
            use_pos = row_pos
            qlbl = query_pred_labels[i]
            if qlbl != "unassigned":
                match = ref_labels[use_idx] == qlbl
                if np.any(match):
                    use_idx = use_idx[match]
                    use_dist = use_dist[match]
                    if use_pos is not None:
                        use_pos = use_pos[match]
            if use_idx.size == 0:
                continue
            if proj_mode == "medoid":
                dist_use = np.where(np.isfinite(use_dist), use_dist, np.inf)
                if rng_jitter is not None:
                    dist_use = dist_use + rng_jitter.normal(0.0, medoid_jitter, size=dist_use.shape)
                top_m = int(medoid_topm) if medoid_topm is not None else 1
                top_m = max(1, min(top_m, use_idx.size))
                if top_m == 1:
                    pick_local = int(np.argmin(dist_use))
                else:
                    cand_local = np.argpartition(dist_use, top_m - 1)[:top_m]
                    cand_idx = use_idx[cand_local]
                    cand_latent = ref_latent[cand_idx]
                    diff = cand_latent[:, None, :] - cand_latent[None, :, :]
                    dsum = np.einsum("ijk,ijk->ij", diff, diff).sum(axis=1)
                    pick_local = int(cand_local[np.argmin(dsum)])
                if use_pos is not None:
                    coords[i] = ref_adata.obsm["X_umap"][use_pos[pick_local]]
                else:
                    coords[i] = ref_umap[use_idx[pick_local]]
            else:
                if weight_mode == "softmax":
                    if tau_quantile > 0:
                        tau_q = np.quantile(use_dist, tau_quantile)
                        tau_q = max(tau_q, 1e-8)
                        weights = np.exp(-use_dist / tau_q)
                    elif tau <= 0:
                        tau_vec = np.median(use_dist)
                        tau_vec = max(tau_vec, 1e-8)
                        weights = np.exp(-use_dist / tau_vec)
                    else:
                        weights = np.exp(-use_dist / max(tau, 1e-8))
                else:
                    weights = 1.0 / (use_dist + 1e-8)
                wsum = weights.sum()
                if wsum <= 0:
                    continue
                weights = weights / wsum
                if use_pos is not None:
                    ref_neighbors = ref_adata.obsm["X_umap"][use_pos]
                else:
                    ref_neighbors = ref_umap[use_idx]
                coords[i] = (weights[:, None] * ref_neighbors).sum(axis=0)
        return coords

    if proj_mode == "medoid":
        dist_use = np.where(np.isfinite(dist), dist, np.inf)
        if medoid_jitter > 0:
            rng = np.random.default_rng(medoid_jitter_seed)
            dist_use = dist_use + rng.normal(0.0, medoid_jitter, size=dist_use.shape)
        top_m = int(medoid_topm) if medoid_topm is not None else 1
        top_m = max(1, min(top_m, k))
        if top_m == 1:
            medoid_local = np.argmin(dist_use, axis=1)
        else:
            medoid_local = np.empty(n_query, dtype=int)
            for i in range(n_query):
                row = dist_use[i]
                cand_local = np.argpartition(row, top_m - 1)[:top_m]
                cand_idx = idx[i, cand_local]
                cand_latent = ref_latent[cand_idx]
                diff = cand_latent[:, None, :] - cand_latent[None, :, :]
                dsum = np.einsum("ijk,ijk->ij", diff, diff).sum(axis=1)
                medoid_local[i] = cand_local[np.argmin(dsum)]
        if ref_adata is not None and nbr_pos is not None:
            best_pos = nbr_pos[np.arange(n_query), medoid_local]
            return ref_adata.obsm["X_umap"][best_pos]
        best_idx = idx[np.arange(n_query), medoid_local]
        return ref_umap[best_idx]
    if weight_mode == "softmax":
        if tau_quantile > 0:
            tau_q = np.quantile(dist, tau_quantile, axis=1, keepdims=True)
            tau_q = np.maximum(tau_q, 1e-8)
            weights = np.exp(-dist / tau_q)
        elif tau <= 0:
            tau_vec = np.median(dist, axis=1, keepdims=True)
            tau_vec = np.maximum(tau_vec, 1e-8)
            weights = np.exp(-dist / tau_vec)
        else:
            weights = np.exp(-dist / max(tau, 1e-8))
    else:
        weights = 1.0 / (dist + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    if ref_adata is not None and nbr_pos is not None:
        ref_neighbors = ref_adata.obsm["X_umap"][nbr_pos]
    else:
        ref_neighbors = ref_umap[idx]
    return (weights[:, :, None] * ref_neighbors).sum(axis=1)


def maybe_umap_overlay(
    ref_latent: np.ndarray,
    query_latent: np.ndarray,
    query_labels: np.ndarray,
    query_ids: np.ndarray | None,
    out_dir: str,
    seed: int,
    max_ref: int,
    query_label_col: str,
    use_precomputed: bool,
    ref_umap: np.ndarray | None,
    ref_adata: ad.AnnData | None,
    knn_k: int,
    umap_proj_mode: str,
    umap_medoid_topm: int,
    umap_medoid_jitter: float,
    umap_plot_jitter: float,
    umap_query_max_dist: float,
    umap_query_dist_quantile: float,
    ref_labels: np.ndarray | None,
    umap_weight: str,
    umap_tau: float,
    umap_tau_quantile: float,
    umap_filter_quantile: float,
    query_keep_mask: np.ndarray | None,
    label_palette: dict[str, str] | None,
    query_color_labels: np.ndarray | None,
    color_palette: dict[str, str] | None,
    umap_proj_label_filter: bool = False,
    umap_proj_labels: np.ndarray | None = None,
    umap_knn_idx: np.ndarray | None = None,
    umap_knn_dist: np.ndarray | None = None,
    umap_knn_ref_latent: np.ndarray | None = None,
    umap_knn_ref_labels: np.ndarray | None = None,
    umap_knn_ref_ids: np.ndarray | None = None,
) -> np.ndarray | None:
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    query_idx = np.arange(query_latent.shape[0])
    if use_precomputed:
        if ref_umap is None:
            raise ValueError("Missing precomputed UMAP coordinates.")
        proj_ref_latent = ref_latent
        proj_ref_umap = ref_umap
        proj_ref_adata = ref_adata
        proj_knn_idx = None
        proj_knn_dist = None
        proj_ref_labels = ref_labels
        proj_query_labels = umap_proj_labels
        if umap_proj_label_filter:
            if umap_proj_labels is None:
                raise ValueError("UMAP label filter enabled without query labels.")
            if (
                umap_knn_idx is not None
                and umap_knn_dist is not None
                and umap_knn_ref_latent is not None
                and umap_knn_ref_labels is not None
                and umap_knn_ref_ids is not None
                and ref_adata is not None
            ):
                ref_ids = ref_adata.obs_names.astype(str).to_numpy()
                ref_pos = {cid: i for i, cid in enumerate(ref_ids)}
                knn_pos = np.array([ref_pos.get(cid, -1) for cid in umap_knn_ref_ids], dtype=int)
                if np.any(knn_pos < 0):
                    print("UMAP label-proj fallback: missing ref ids in umap ref.")
                else:
                    proj_ref_umap = ref_adata.obsm["X_umap"][knn_pos]
                    proj_ref_latent = umap_knn_ref_latent
                    proj_ref_adata = None
                    proj_knn_idx = umap_knn_idx
                    proj_knn_dist = umap_knn_dist
                    proj_ref_labels = umap_knn_ref_labels
        query_coords = project_query_umap_knn(
            proj_ref_latent,
            proj_ref_umap,
            query_latent,
            knn_k,
            umap_weight,
            umap_tau,
            umap_tau_quantile,
            ref_adata=proj_ref_adata,
            proj_mode=umap_proj_mode,
            medoid_topm=umap_medoid_topm,
            medoid_jitter=umap_medoid_jitter,
            medoid_jitter_seed=seed,
            knn_idx=proj_knn_idx,
            knn_dist=proj_knn_dist,
            label_filter=umap_proj_label_filter,
            ref_labels=proj_ref_labels,
            query_pred_labels=proj_query_labels,
        )
        ref_coords = ref_umap
    else:
        if not _HAS_UMAP:
            return
        if ref_latent.shape[0] > max_ref:
            keep = rng.choice(ref_latent.shape[0], size=max_ref, replace=False)
            ref_latent = ref_latent[keep]

        combined = np.vstack([ref_latent, query_latent])
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.5, random_state=seed)
        coords = reducer.fit_transform(combined)
        ref_coords = coords[: ref_latent.shape[0]]
        query_coords = coords[ref_latent.shape[0] :]

    if (not use_precomputed) and ref_coords.shape[0] > max_ref:
        keep = rng.choice(ref_coords.shape[0], size=max_ref, replace=False)
        ref_coords = ref_coords[keep]

    if umap_query_dist_quantile and not (0.0 <= umap_query_dist_quantile <= 1.0):
        raise ValueError("--umap-query-dist-quantile must be in [0, 1].")
    if (umap_query_max_dist and umap_query_max_dist > 0) or (
        umap_query_dist_quantile and umap_query_dist_quantile > 0
    ):
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(ref_latent)
        dist_gate, _ = nn.kneighbors(query_latent, return_distance=True)
        dist_gate = dist_gate[:, 0]
        keep_gate = np.ones_like(dist_gate, dtype=bool)
        if umap_query_max_dist and umap_query_max_dist > 0:
            keep_gate &= dist_gate <= umap_query_max_dist
        if umap_query_dist_quantile and umap_query_dist_quantile > 0:
            thr = float(np.quantile(dist_gate, umap_query_dist_quantile))
            keep_gate &= dist_gate <= thr
        kept = int(np.sum(keep_gate))
        print(
            "UMAP query distance gate: max_dist={md} quantile={q} "
            "kept={k}/{n}".format(
                md=umap_query_max_dist,
                q=umap_query_dist_quantile,
                k=kept,
                n=keep_gate.size,
            )
        )
        if query_keep_mask is None:
            query_keep_mask = keep_gate
        else:
            if query_keep_mask.shape[0] != keep_gate.shape[0]:
                raise ValueError("Query keep mask size mismatch.")
            query_keep_mask = query_keep_mask & keep_gate

    if query_ids is not None:
        query_ids = np.asarray(query_ids)

    if query_keep_mask is not None:
        if query_keep_mask.shape[0] != query_coords.shape[0]:
            raise ValueError("UMAP keep mask size mismatch.")
        query_idx = query_idx[query_keep_mask]
        query_coords = query_coords[query_keep_mask]
        query_labels = query_labels[query_keep_mask]
        if query_color_labels is not None:
            query_color_labels = query_color_labels[query_keep_mask]
        if query_ids is not None:
            query_ids = query_ids[query_keep_mask]
    if umap_filter_quantile and umap_filter_quantile > 0 and query_coords.shape[0] > 1:
        k = min(20, query_coords.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(query_coords)
        dists, _ = nn.kneighbors(query_coords)
        mean_dist = dists[:, 1:].mean(axis=1)
        thr = float(np.quantile(mean_dist, umap_filter_quantile))
        keep = mean_dist <= thr
        kept = int(np.sum(keep))
        print(
            f"UMAP coord filter: quantile={umap_filter_quantile} "
            f"thr={thr:.4f} kept={kept}/{keep.size}"
        )
        query_idx = query_idx[keep]
        query_coords = query_coords[keep]
        query_labels = query_labels[keep]
        if query_color_labels is not None:
            query_color_labels = query_color_labels[keep]
        if query_ids is not None:
            query_ids = query_ids[keep]
    labels_ser = pd.Series(query_labels).astype(str)
    drop_mask = labels_ser.str.lower().isin({"unassigned", "unknown", "nan"})
    keep_mask = ~drop_mask
    if not np.all(keep_mask):
        query_idx = query_idx[keep_mask]
        query_coords = query_coords[keep_mask]
        query_labels = labels_ser.to_numpy()[keep_mask]
        if query_color_labels is not None:
            query_color_labels = query_color_labels[keep_mask]
        if query_ids is not None:
            query_ids = query_ids[keep_mask]
    else:
        query_labels = labels_ser.to_numpy()
    if query_color_labels is not None:
        query_color_labels = pd.Series(query_color_labels).astype(str).to_numpy()

    ref_df = pd.DataFrame(ref_coords, columns=["umap1", "umap2"])
    query_df = pd.DataFrame(
        {
            "umap1": query_coords[:, 0],
            "umap2": query_coords[:, 1],
            query_label_col: query_labels,
        }
    )
    if query_ids is not None:
        query_df.insert(0, "cell_id", query_ids.astype(str))
    ref_df.to_csv(os.path.join(out_dir, "ref_umap.csv"), index=False)
    query_df.to_csv(os.path.join(out_dir, "query_umap.csv"), index=False)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from matplotlib import colors as mcolors
        try:
            from adjustText import adjust_text  # type: ignore
            _HAS_ADJUST = True
        except Exception:
            _HAS_ADJUST = False

        plot_query_coords = query_coords
        if umap_plot_jitter and umap_plot_jitter > 0:
            rng_plot = np.random.default_rng(seed + 101)
            plot_query_coords = query_coords + rng_plot.normal(
                0.0, umap_plot_jitter, size=query_coords.shape
            )

        def build_color_map(labels: np.ndarray, base_palette: dict[str, str] | None):
            label_list = pd.unique(labels)
            if base_palette:
                return {lbl: base_palette.get(lbl, "#7f7f7f") for lbl in label_list}
            if len(label_list) <= 20:
                cmap = plt.get_cmap("tab20")
                return {lbl: cmap(i % 20) for i, lbl in enumerate(label_list)}
            cmap = plt.get_cmap("hsv")
            denom = max(1, len(label_list) - 1)
            return {lbl: cmap(i / denom) for i, lbl in enumerate(label_list)}

        def density_weights(coords: np.ndarray, k: int = 20) -> np.ndarray:
            if coords.shape[0] == 0:
                return np.array([])
            if coords.shape[0] <= 2:
                return np.ones(coords.shape[0], dtype=float)
            k = min(k, coords.shape[0] - 1)
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(coords)
            dists, _ = nn.kneighbors(coords)
            mean_dist = dists[:, 1:].mean(axis=1)
            inv = 1.0 / (mean_dist + 1e-6)
            lo = np.quantile(inv, 0.05)
            hi = np.quantile(inv, 0.95)
            if hi <= lo:
                return np.ones_like(inv)
            w = (inv - lo) / (hi - lo)
            return np.clip(w, 0.0, 1.0)

        def pick_dense_anchor(coords: np.ndarray) -> np.ndarray | None:
            if coords.shape[0] == 0:
                return None
            centroid = coords.mean(axis=0)
            k = min(30, max(5, int(coords.shape[0] * 0.05)))
            if coords.shape[0] <= k:
                idx = ((coords - centroid) ** 2).sum(axis=1).argmin()
                return coords[idx]
            nn = NearestNeighbors(n_neighbors=min(k + 1, coords.shape[0]))
            nn.fit(coords)
            dists, _ = nn.kneighbors(coords)
            mean_dist = dists[:, 1:].mean(axis=1)
            return coords[mean_dist.argmin()]

        fig, ax = plt.subplots(figsize=(12, 9))
        ref_size = 1
        query_size = 3
        ax.scatter(
            ref_coords[:, 0],
            ref_coords[:, 1],
            s=ref_size,
            c="#bdbdbd",
            alpha=0.4,
            linewidths=0,
            label="reference",
            rasterized=True,
        )
        label_order = pd.unique(query_labels)
        color_map = build_color_map(label_order, label_palette)
        dens_w = density_weights(plot_query_coords, k=15)
        if dens_w.size == 0:
            dens_w = np.ones(plot_query_coords.shape[0], dtype=float)
        else:
            thr = np.quantile(dens_w, 0.1)
            dens_w = np.clip(dens_w, thr, 1.0)
        point_sizes = 0.5 + 0.5 * dens_w
        point_alphas = 0.7 + 0.15 * dens_w
        if query_color_labels is not None and color_palette:
            query_colors = [color_palette.get(lbl, "#1f77b4") for lbl in query_color_labels]
            query_rgba = [
                mcolors.to_rgba(c, alpha=a) for c, a in zip(query_colors, point_alphas)
            ]
            ax.scatter(
                plot_query_coords[:, 0],
                plot_query_coords[:, 1],
                s=point_sizes,
                linewidths=0,
                c=query_rgba,
                rasterized=True,
            )
            label_color_map = {}
            for lbl in label_order:
                mask = query_labels == lbl
                if not np.any(mask):
                    continue
                vals, counts = np.unique(query_color_labels[mask], return_counts=True)
                pick = vals[np.argmax(counts)]
                label_color_map[lbl] = color_palette.get(pick, "#1f77b4")
        else:
            label_color_map = color_map
            query_colors = [label_color_map.get(lbl, "#1f77b4") for lbl in query_labels]
            query_rgba = [
                mcolors.to_rgba(c, alpha=a) for c, a in zip(query_colors, point_alphas)
            ]
            ax.scatter(
                plot_query_coords[:, 0],
                plot_query_coords[:, 1],
                s=point_sizes,
                linewidths=0,
                c=query_rgba,
                rasterized=True,
            )
        ax.set_title("")
        ax.set_axis_off()
        texts = []
        anchors = {}
        x_span = float(np.max(ref_coords[:, 0]) - np.min(ref_coords[:, 0]))
        y_span = float(np.max(ref_coords[:, 1]) - np.min(ref_coords[:, 1]))
        radius = 0.04 * max(x_span, y_span)
        max_label_dist = 0.08 * max(x_span, y_span)
        labels = [lbl for lbl in label_order if np.any(query_labels == lbl)]
        for lbl in labels:
            mask = query_labels == lbl
            if not np.any(mask):
                continue
            coords = plot_query_coords[mask]
            anchor = pick_dense_anchor(coords)
            if anchor is None:
                continue
            anchors[lbl] = anchor
        for i, lbl in enumerate(labels):
            anchor = anchors.get(lbl)
            if anchor is None:
                continue
            angle = 2.0 * np.pi * (i / max(1, len(labels)))
            tx = anchor[0] + radius * np.cos(angle)
            ty = anchor[1] + radius * np.sin(angle)
            label_text = str(lbl).replace("_", "\n")
            txt = ax.text(
                tx,
                ty,
                label_text,
                fontsize=8,
                color=label_color_map.get(lbl, "#1f77b4"),
                ha="center",
                va="center",
            )
            txt.set_path_effects([pe.withStroke(linewidth=1.0, foreground="white")])
            texts.append((txt, lbl))
        if _HAS_ADJUST and texts:
            adjust_text(
                [t[0] for t in texts],
                ax=ax,
                expand=(2.0, 2.4),
                force_points=(0.8, 1.0),
                force_text=(1.2, 1.6),
                only_move={"points": "xy", "text": "xy"},
                lim=1000,
            )
        for txt, lbl in texts:
            anchor = anchors.get(lbl)
            if anchor is None:
                continue
            x, y = txt.get_position()
            dx = x - anchor[0]
            dy = y - anchor[1]
            dist = float(np.hypot(dx, dy))
            if dist > max_label_dist and dist > 0:
                scale = max_label_dist / dist
                txt.set_position((anchor[0] + dx * scale, anchor[1] + dy * scale))
        for txt, lbl in texts:
            anchor = anchors.get(lbl)
            if anchor is None:
                continue
            line = ax.plot(
                [anchor[0], txt.get_position()[0]],
                [anchor[1], txt.get_position()[1]],
                color=txt.get_color(),
                lw=1.0,
                alpha=0.9,
                zorder=4,
                solid_capstyle="round",
            )[0]
            line.set_path_effects(
                [pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()]
            )
        fig.subplots_adjust(bottom=0.12, left=0.08, right=0.98, top=0.98)
        fig.savefig(os.path.join(out_dir, "encoder_umap_overlay.png"), dpi=600)
        fig.savefig(
            os.path.join(out_dir, "encoder_umap_overlay.pdf"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.scatter(
            ref_coords[:, 0],
            ref_coords[:, 1],
            s=ref_size,
            c="#bdbdbd",
            alpha=0.4,
            linewidths=0,
        )
        ax.scatter(
            plot_query_coords[:, 0],
            plot_query_coords[:, 1],
            s=query_size,
            alpha=0.9,
            linewidths=0,
            color="#1f77b4",
        )
        ax.set_title("")
        ax.set_axis_off()
        fig.subplots_adjust(bottom=0.12, left=0.08, right=0.98, top=0.98)
        fig.savefig(os.path.join(out_dir, "encoder_umap_overlay_blue.png"), dpi=600)
        plt.close(fig)

        if ref_labels is not None and ref_labels.shape[0] == ref_coords.shape[0]:
            fig, ax = plt.subplots(figsize=(12, 9))
            uniq = pd.unique(ref_labels)
            color_map = build_color_map(uniq, label_palette)
            for lbl in uniq:
                mask = ref_labels == lbl
                ax.scatter(
                    ref_coords[mask, 0],
                    ref_coords[mask, 1],
                    s=ref_size,
                    alpha=0.7,
                    linewidths=0,
                    color=color_map.get(lbl, "#1f77b4"),
                )
            ax.set_title("")
            ax.set_axis_off()
            fig.subplots_adjust(bottom=0.12, left=0.08, right=0.98, top=0.98)
            fig.savefig(
                os.path.join(out_dir, "reference_umap_colored.png"), dpi=600
            )
            plt.close(fig)
    except Exception:
        pass
    full_coords = np.full((query_latent.shape[0], 2), np.nan, dtype=np.float32)
    if query_coords.size:
        full_coords[query_idx] = query_coords
    return full_coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Train encoder and project query.")
    parser.add_argument(
        "--ref-path",
        default="not_bundled/ZMAP_251209_processed.h5ad",  #change path
    )
    parser.add_argument(
        "--train-ref-path",
        default=None,
        help="Optional subset reference path for training only.",
    )
    parser.add_argument(
        "--val-ref-path",
        default=None,
        help="Optional separate reference path used only for validation.",
    )
    parser.add_argument(
        "--train-ids",
        default=None,
        help="Optional CSV of cell_ids to restrict training/ref pool (subset of ref.obs_names).",
    )
    parser.add_argument(
        "--query-path",
        default="not_bundled/ZMAP_251209_processed_Sur2023_frac30_geosketch.h5ad",  #change path
    )
    parser.add_argument(
        "--exclude-path",
        default="not_bundled/ZMAP_251209_processed_Sur2023_with_selected.h5ad",  #change path
        help="Path with geosketch_selected to exclude from reference.",
    )
    parser.add_argument("--exclude-col", default="geosketch_selected")
    parser.add_argument(
        "--exclude-ids-csv",
        default=None,
        help="Optional CSV of cell IDs to exclude from ref/train in addition to --exclude-path/query IDs.",
    )
    parser.add_argument("--hvg-key", default="highly_variable")
    parser.add_argument("--latent-key", default="X_pca")
    parser.add_argument("--layer-key", default="raw_nolog")
    parser.add_argument(
        "--input-normalize",
        action="store_true",
        help="Normalize counts per cell before log1p (normalize_total).",
    )
    parser.add_argument(
        "--input-target-sum",
        type=float,
        default=1e6,
        help="Target sum for input normalization.",
    )
    parser.add_argument(
        "--input-log1p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply log1p after optional normalize_total preprocessing.",
    )
    parser.add_argument(
        "--input-scale",
        action="store_true",
        help="Scale input genes using reference mean/std after normalization/log1p.",
    )
    parser.add_argument("--label-col", default="ZMAP_CellType")
    parser.add_argument("--coarse-label-col", default="ZMAP_GermLayer")
    parser.add_argument("--latent-scale", choices=["none", "zscore"], default="zscore")
    parser.add_argument("--coarse-loss-weight", type=float, default=0.1)
    parser.add_argument("--coarse-margin", type=float, default=1.0)
    parser.add_argument("--coarse-subsample", type=int, default=2048)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument(
        "--hidden",
        nargs="*",
        type=int,
        default=[1024, 512],
        help="Hidden layer sizes before the latent output. Pass no values for direct HVG->latent.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--use-layernorm",
        action="store_true",
        help="Deprecated alias for --input-layernorm.",
    )
    parser.add_argument(
        "--input-layernorm",
        action="store_true",
        help="Apply LayerNorm to HVG input features before the MLP.",
    )
    parser.add_argument(
        "--no-hidden-layernorm",
        action="store_true",
        help="Disable LayerNorm in hidden MLP blocks (keep only input LayerNorm if enabled).",
    )
    parser.add_argument(
        "--mse-by-label",
        action="store_true",
        help="Compute reconstruction loss as mean MSE within coarse labels.",
    )
    parser.add_argument(
        "--exclude-unassigned-train",
        action="store_true",
        help="Exclude unassigned cells from training batches.",
    )
    parser.add_argument(
        "--query-full-layer",
        default=None,
        help="Use query raw or layer to fill missing HVG genes (set to 'raw' for adata.raw).",
    )
    parser.add_argument(
        "--tissue-col",
        default=None,
        help="Tissue column used to condition kNN label transfer.",
    )
    parser.add_argument(
        "--tissue-conditioned",
        action="store_true",
        help="Condition kNN label transfer on tissue (if available).",
    )
    parser.add_argument(
        "--tissue-aware-mode",
        choices=["hard", "soft"],
        default="hard",
        help="kNN tissue-aware mode when --tissue-conditioned is set.",
    )
    parser.add_argument(
        "--tissue-penalty-lambda",
        type=float,
        default=1.0,
        help="Penalty weight for soft tissue-aware kNN.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--residual-blocks",
        type=int,
        default=0,
        help="Number of residual blocks after hidden layers (0 = disabled).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-frac", type=float, default=0.0)
    parser.add_argument("--min-lr-factor", type=float, default=0.0)
    parser.add_argument("--cosine-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--domain-method",
        choices=["auto", "none", "coral", "dann", "mmd"],
        default="auto",
        help="Domain alignment method for query regularization.",
    )
    parser.add_argument(
        "--domain-coral-weight",
        type=float,
        default=0.0,
        help="Weight for CORAL domain alignment loss between reference/query encoder outputs.",
    )
    parser.add_argument(
        "--domain-coral-mean-weight",
        type=float,
        default=1.0,
        help="Extra weight for mean-alignment term inside CORAL loss.",
    )
    parser.add_argument(
        "--domain-dann-weight",
        type=float,
        default=0.0,
        help="Weight for DANN adversarial domain loss.",
    )
    parser.add_argument(
        "--domain-dann-grl-coeff",
        type=float,
        default=1.0,
        help="Gradient reversal coefficient for DANN.",
    )
    parser.add_argument(
        "--domain-dann-hidden",
        default="256,128",
        help="Comma-separated hidden sizes for DANN domain classifier.",
    )
    parser.add_argument(
        "--domain-dann-dropout",
        type=float,
        default=0.1,
        help="Dropout for DANN domain classifier.",
    )
    parser.add_argument(
        "--domain-mmd-weight",
        type=float,
        default=0.0,
        help="Weight for MMD domain alignment loss.",
    )
    parser.add_argument(
        "--domain-mmd-kernel",
        choices=["rbf", "linear"],
        default="rbf",
        help="Kernel type for MMD loss.",
    )
    parser.add_argument(
        "--domain-mmd-gamma",
        type=float,
        default=0.0,
        help="RBF gamma for MMD (<=0 means infer from median distance).",
    )
    parser.add_argument(
        "--domain-batch-size",
        type=int,
        default=0,
        help="Query batch size for domain alignment (0 = use --batch-size).",
    )
    parser.add_argument(
        "--domain-max-query",
        type=int,
        default=0,
        help="Max query cells sampled for domain pool (0 = use all query cells).",
    )
    parser.add_argument("--gene-dropout", type=float, default=0.0)
    parser.add_argument(
        "--module-target-markers-csv",
        default=None,
        help="CSV of module marker genes (expects gene/celltype, optional module_level/module_weight).",
    )
    parser.add_argument(
        "--module-min-genes",
        type=int,
        default=3,
        help="Minimum HVG-overlapping genes required to keep a module.",
    )
    parser.add_argument(
        "--lambda-module-loss",
        type=float,
        default=0.0,
        help="Weight for auxiliary module reconstruction loss from latent.",
    )
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-libsize", action="store_true")
    parser.add_argument("--cache-hvg", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--out-root", default="not_bundled/output/phase1")  #change path
    parser.add_argument(
        "--train-max-ref",
        type=int,
        default=0,
        help="Max reference cells to use for training/validation (0 = use all).",
    )
    parser.add_argument(
        "--mask-hvg-frac",
        type=float,
        default=0.0,
        help="Mask this fraction of HVG genes to zero in input (0 = disabled).",
    )
    parser.add_argument(
        "--mask-hvg-seed",
        type=int,
        default=0,
        help="Random seed for HVG masking (0 = use --seed).",
    )
    parser.add_argument(
        "--umap-ref-path",
        default=None,
        help="Optional full-atlas path for UMAP background only.",
    )
    parser.add_argument(
        "--umap-use-precomputed",
        action="store_true",
        help="Use precomputed X_umap for background and project query by kNN.",
    )
    parser.add_argument(
        "--umap-knn-k",
        type=int,
        default=0,
        help="kNN size for UMAP projection when using precomputed UMAP.",
    )
    parser.add_argument("--umap-max-ref", type=int, default=200000)
    parser.add_argument("--no-umap", action="store_true")
    parser.add_argument(
        "--umap-weight",
        choices=["inv_dist", "softmax"],
        default="inv_dist",
        help="UMAP projection weight mode (softmax uses exp(-d/tau)).",
    )
    parser.add_argument(
        "--umap-proj-mode",
        choices=["mean", "medoid"],
        default="medoid",
        help="UMAP projection mode (weighted mean or medoid neighbor).",
    )
    parser.add_argument(
        "--umap-proj-label-filter",
        action="store_true",
        help="Filter UMAP projection neighbors to match predicted label (label-transfer kNN).",
    )
    parser.add_argument(
        "--umap-medoid-topm",
        type=int,
        default=1,
        help="For medoid projection: choose the medoid among top-m nearest neighbors (1 = nearest).",
    )
    parser.add_argument(
        "--umap-medoid-jitter",
        type=float,
        default=0.0,
        help="For medoid projection: add Gaussian jitter to distances (0 = disabled).",
    )
    parser.add_argument(
        "--umap-plot-jitter",
        type=float,
        default=0.0,
        help="Add Gaussian jitter to query UMAP coords for plotting only (0 = disabled).",
    )
    parser.add_argument(
        "--umap-query-max-dist",
        type=float,
        default=0.0,
        help="Gate query points by nearest-neighbor distance in latent space (0 = disabled).",
    )
    parser.add_argument(
        "--umap-query-dist-quantile",
        type=float,
        default=0.0,
        help="Gate query points by nearest-neighbor distance quantile in latent space (0 = disabled).",
    )
    parser.add_argument(
        "--umap-tau",
        type=float,
        default=0.0,
        help="Softmax temperature for UMAP projection; <=0 uses per-query median distance.",
    )
    parser.add_argument(
        "--umap-tau-quantile",
        type=float,
        default=0.0,
        help="Use per-query distance quantile for softmax tau (0 = disabled).",
    )
    parser.add_argument(
        "--umap-filter-quantile",
        type=float,
        default=0.95,
        help="Drop query points above this knn_mean_dist quantile in UMAP plots (0 = disabled).",
    )
    parser.add_argument(
        "--latent-filter-label",
        default=None,
        help="Reference label to filter query cells by distance in latent space (omit to use all reference cells).",
    )
    parser.add_argument(
        "--latent-filter-metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric for latent filter (cosine or euclidean).",
    )
    parser.add_argument(
        "--latent-filter-quantile",
        type=float,
        default=0.0,
        help="Keep query points within this distance quantile to the reference label (0 = disabled).",
    )
    parser.add_argument(
        "--latent-filter-k",
        type=int,
        default=50,
        help="Number of reference label neighbors for mean distance (kNN).",
    )
    parser.add_argument("--include-unassigned", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--knn-metric",
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Distance metric for kNN in latent space.",
    )
    parser.add_argument(
        "--knn-l2norm",
        action="store_true",
        help="L2 normalize latent vectors before kNN (recommended for cosine).",
    )
    parser.add_argument(
        "--omit-eval-labels",
        default="",
        help="Comma-separated true labels to exclude from evaluation metrics.",
    )
    parser.add_argument(
        "--time-col",
        default=None,
        help="Categorical time column for kNN regression (e.g., time_block_id).",
    )
    parser.add_argument(
        "--time-order",
        default=None,
        help="Comma-separated ordered time categories for ordinal encoding.",
    )
    parser.add_argument(
        "--time-topk",
        type=int,
        default=0,
        help="Keep top-k weights for time regression (0 = use all k).",
    )
    parser.add_argument(
        "--time-hard-topk",
        type=int,
        default=0,
        help="Top-k neighbors for hard time label voting (0 = use all k).",
    )
    parser.add_argument(
        "--time-monotone-delta",
        type=int,
        default=0,
        help="Ordinal delta for monotone downweighting (0 = disabled).",
    )
    parser.add_argument(
        "--time-monotone-gamma",
        type=float,
        default=1.0,
        help="Downweight factor for earlier-than-observed neighbors (1 = disabled).",
    )
    parser.add_argument(
        "--time-trim-extremes",
        type=int,
        default=0,
        help="Trim N smallest and largest time values after weighting (0 = disabled).",
    )
    args = parser.parse_args()
    # Backward-compatible alias: --use-layernorm == --input-layernorm.
    args.input_layernorm = bool(args.input_layernorm or args.use_layernorm)
    # Keep legacy key in config for downstream scripts that still read use_layernorm.
    args.use_layernorm = bool(args.input_layernorm)
    args.hidden_layernorm = not bool(args.no_hidden_layernorm)
    if int(args.residual_blocks) < 0:
        raise ValueError("--residual-blocks must be >= 0.")
    if bool(args.input_log1p) and str(args.layer_key).lower().endswith("log1p"):
        raise ValueError(
            "--input-log1p is enabled but --layer-key already appears log-transformed. "
            "Use --no-input-log1p with a log1p layer."
        )

    domain_weights = {
        "coral": float(args.domain_coral_weight),
        "dann": float(args.domain_dann_weight),
        "mmd": float(args.domain_mmd_weight),
    }
    explicit_domain_method = str(args.domain_method).lower()
    if explicit_domain_method == "auto":
        active = [k for k, v in domain_weights.items() if v > 0]
        if len(active) > 1:
            raise ValueError(
                "Multiple domain weights are > 0 under --domain-method auto. "
                "Set one weight > 0 or set --domain-method explicitly."
            )
        domain_method = active[0] if active else "none"
    else:
        domain_method = explicit_domain_method

    if domain_method == "none":
        domain_weight = 0.0
    else:
        domain_weight = float(domain_weights[domain_method])
        if domain_weight <= 0:
            raise ValueError(
                f"--domain-method {domain_method} requires positive matching domain weight."
            )
    args.domain_method_resolved = domain_method

    ensure_dir(args.out_root)
    subdirs = [
        "model",
        "projection",
        "label_transfer",
        "metrics",
        "runtime",
        "umap",
        "time_regression",
    ]
    for sd in subdirs:
        ensure_dir(os.path.join(args.out_root, sd))

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    adata_ref_full = ad.read_h5ad(args.ref_path, backed="r")
    adata_ref_train = adata_ref_full
    if args.train_ref_path:
        adata_ref_train = ad.read_h5ad(args.train_ref_path, backed="r")
    adata_ref_val = None
    if args.val_ref_path:
        adata_ref_val = ad.read_h5ad(args.val_ref_path, backed="r")
    adata_query = ad.read_h5ad(args.query_path, backed="r")
    adata_umap_ref = adata_ref_full
    if args.umap_ref_path:
        adata_umap_ref = ad.read_h5ad(args.umap_ref_path, backed="r")

    if args.hvg_key not in adata_ref_full.var:
        raise KeyError(f"Missing HVG key in reference: {args.hvg_key}")
    hvg_genes = hvg_genes_from_key(adata_ref_full, args.hvg_key)
    hvg_idx_ref = map_genes_to_indices(adata_ref_full.var_names, hvg_genes, "reference")
    hvg_idx_train = map_genes_to_indices(
        adata_ref_train.var_names, hvg_genes, "train reference"
    )
    hvg_idx_val = hvg_idx_train
    if adata_ref_val is not None:
        hvg_idx_val = map_genes_to_indices(
            adata_ref_val.var_names, hvg_genes, "validation reference"
        )
    query_index = pd.Index(adata_query.var_names.astype(str))
    hvg_names = np.array(hvg_genes, dtype=object)
    query_hvg_indexer = query_index.get_indexer(hvg_names)
    present_mask = query_hvg_indexer >= 0
    present_query_idx = query_hvg_indexer[present_mask].astype(int)
    hvg_present_pos = np.nonzero(present_mask)[0]
    missing_mask = ~present_mask

    query_full = None
    query_full_index = None
    full_layer_key = None
    query_full_layer = args.query_full_layer
    if query_full_layer:
        if query_full_layer == "raw":
            if adata_query.raw is None:
                print("query-full-layer=raw requested but query.raw is None.")
            else:
                query_full = adata_query.raw
                query_full_index = pd.Index(query_full.var_names.astype(str))
                full_layer_key = "X"
        else:
            if query_full_layer != "X" and query_full_layer not in adata_query.layers:
                raise KeyError(f"Missing query layer: {query_full_layer}")
            query_full = adata_query
            query_full_index = query_index
            full_layer_key = query_full_layer
    present_full_idx = np.array([], dtype=int)
    hvg_full_pos = np.array([], dtype=int)
    missing_pos = np.array([], dtype=int)
    full_present_mask = None
    if missing_mask.any() and query_full is not None:
        missing_names = hvg_names[missing_mask]
        full_indexer = query_full_index.get_indexer(missing_names)
        full_present_mask = full_indexer >= 0
        present_full_idx = full_indexer[full_present_mask].astype(int)
        missing_pos = np.nonzero(missing_mask)[0]
        hvg_full_pos = missing_pos[full_present_mask]
    if missing_mask.any() and full_present_mask is not None:
        missing_pos = missing_pos[~full_present_mask]
    elif missing_mask.any():
        missing_pos = np.nonzero(missing_mask)[0]
    if missing_mask.any():
        missing_total = int(missing_mask.sum())
        filled_from_full = int(hvg_full_pos.size)
        still_missing = missing_total - filled_from_full
        if still_missing > 0:
            print(
                f"Missing HVG genes in query after full-layer lookup: {still_missing} / {len(hvg_names)} "
                "(filled with zeros)"
            )
    hvg_idx = hvg_idx_ref
    if args.latent_key not in adata_umap_ref.obsm:
        raise ValueError("UMAP reference missing latent key.")
    if args.umap_use_precomputed and "X_umap" not in adata_umap_ref.obsm:
        raise ValueError("UMAP reference missing X_umap.")

    mask_hvg = None
    if args.mask_hvg_frac > 0:
        if not (0 < args.mask_hvg_frac < 1):
            raise ValueError("--mask-hvg-frac must be between 0 and 1.")
        mask_seed = args.mask_hvg_seed if args.mask_hvg_seed > 0 else args.seed
        mask_rng = np.random.default_rng(mask_seed)
        n_mask = int(round(args.mask_hvg_frac * len(hvg_genes)))
        n_mask = min(max(n_mask, 1), len(hvg_genes))
        mask_pos = mask_rng.choice(len(hvg_genes), size=n_mask, replace=False)
        mask_hvg = np.zeros(len(hvg_genes), dtype=bool)
        mask_hvg[mask_pos] = True
        mask_genes = np.array(hvg_genes, dtype=object)[mask_pos]
        mask_path = os.path.join(args.out_root, "model", "mask_hvg_genes.txt")
        with open(mask_path, "w") as f:
            f.write("\n".join(mask_genes))

    tissue_col = args.tissue_col or args.coarse_label_col
    tissue_candidates = []
    if tissue_col:
        tissue_candidates.append(tissue_col)
    tissue_candidates += ["ZMAP_Tissue", "ZMAP_tissue"]
    tissue_col = None
    for col in tissue_candidates:
        if col in adata_ref_full.obs and col in adata_query.obs:
            tissue_col = col
            break
    if tissue_col is None:
        print("Tissue column not found in both ref/query; using global kNN.")
    elif args.tissue_conditioned:
        print(f"Using tissue-conditioned kNN with column: {tissue_col}")

    ref_obs_names = adata_ref_full.obs_names.astype(str)
    train_obs_names = adata_ref_train.obs_names.astype(str)
    val_obs_names = (
        adata_ref_val.obs_names.astype(str) if adata_ref_val is not None else None
    )
    train_ids = None
    if args.train_ids:
        ids_path = Path(args.train_ids)
        if not ids_path.exists():
            raise FileNotFoundError(f"train-ids not found: {ids_path}")
        ids = read_id_list_csv(ids_path)
        if not ids:
            raise ValueError("train-ids file is empty.")
        train_ids = set(ids)
    exclude_ids = None
    if args.exclude_path:
        selected = read_obs_bool(args.exclude_path, args.exclude_col)
        exclude_idx = read_obs_index(args.exclude_path)
        if selected is None:
            exclude_ids = set(exclude_idx.tolist())
        else:
            exclude_ids = set(exclude_idx[selected].tolist())
    else:
        exclude_ids = set(adata_query.obs_names.astype(str).tolist())

    if args.exclude_ids_csv:
        ex_ids_path = Path(args.exclude_ids_csv)
        if not ex_ids_path.exists():
            raise FileNotFoundError(f"exclude-ids-csv not found: {ex_ids_path}")
        extra_ids = read_id_list_csv(ex_ids_path)
        if not extra_ids:
            raise ValueError("exclude-ids-csv is empty.")
        exclude_ids = set(exclude_ids) if exclude_ids is not None else set()
        exclude_ids.update(extra_ids)
        print(f"Loaded {len(extra_ids)} extra exclude IDs from {ex_ids_path}")

    if exclude_ids:
        keep_mask = ~np.isin(ref_obs_names, list(exclude_ids))
        ref_indices = np.where(keep_mask)[0]
        train_keep_mask = ~np.isin(train_obs_names, list(exclude_ids))
        train_ref_indices = np.where(train_keep_mask)[0]
        if val_obs_names is not None:
            val_keep_mask = ~np.isin(val_obs_names, list(exclude_ids))
            val_ref_indices = np.where(val_keep_mask)[0]
        else:
            val_ref_indices = None
    else:
        ref_indices = np.arange(adata_ref_full.n_obs)
        train_ref_indices = np.arange(adata_ref_train.n_obs)
        val_ref_indices = np.arange(adata_ref_val.n_obs) if adata_ref_val is not None else None

    if train_ids:
        ref_keep = np.isin(ref_obs_names, list(train_ids))
        train_keep = np.isin(train_obs_names, list(train_ids))
        ref_indices = ref_indices[ref_keep[ref_indices]]
        train_ref_indices = train_ref_indices[train_keep[train_ref_indices]]
        if train_ref_indices.size == 0:
            raise ValueError("train-ids did not match any training reference cells.")

    if args.exclude_unassigned_train and args.label_col in adata_ref_train.obs:
        train_labels = adata_ref_train.obs[args.label_col].astype(str).to_numpy()
        assigned_mask = train_labels != "unassigned"
        train_ref_indices = train_ref_indices[assigned_mask[train_ref_indices]]

    train_pool = train_ref_indices
    if args.train_max_ref and args.train_max_ref < len(train_ref_indices):
        train_pool = rng.choice(
            train_ref_indices, size=args.train_max_ref, replace=False
        )
    if adata_ref_val is None:
        rng.shuffle(train_pool)
        n_val = int(len(train_pool) * args.val_frac)
        val_indices = train_pool[:n_val]
        train_indices = train_pool[n_val:]
        print(
            f"Validation split from training reference (--val-frac={args.val_frac}): "
            f"train={len(train_indices)}, val={len(val_indices)}"
        )
    else:
        train_indices = train_pool
        if val_ref_indices is None:
            val_ref_indices = np.arange(adata_ref_val.n_obs)
        val_indices = val_ref_indices
        print(
            f"Using explicit validation reference: train={len(train_indices)}, "
            f"val={len(val_indices)}"
        )
    domain_query_pool = np.arange(adata_query.n_obs, dtype=int)
    if domain_method != "none":
        if int(args.domain_max_query) > 0 and int(args.domain_max_query) < domain_query_pool.size:
            domain_query_pool = rng.choice(
                domain_query_pool, size=int(args.domain_max_query), replace=False
            )
        domain_batch_size = (
            int(args.domain_batch_size)
            if int(args.domain_batch_size) > 0
            else int(args.batch_size)
        )
        if domain_query_pool.size == 0:
            raise ValueError("No query cells available for domain alignment.")
        msg = (
            f"Enabled domain loss: method={domain_method}, lambda={domain_weight:.6f}, "
            f"domain_batch_size={domain_batch_size}, domain_pool={int(domain_query_pool.size)}"
        )
        if domain_method == "coral":
            msg += f", coral_mean_weight={float(args.domain_coral_mean_weight):.6f}"
        elif domain_method == "dann":
            msg += (
                f", grl={float(args.domain_dann_grl_coeff):.6f}, "
                f"clf_hidden={parse_int_list(args.domain_dann_hidden)}"
            )
        elif domain_method == "mmd":
            msg += (
                f", kernel={str(args.domain_mmd_kernel)}, gamma={float(args.domain_mmd_gamma):.6g}"
            )
        print(msg)
    else:
        domain_batch_size = 0

    latent_dim = adata_ref_full.obsm[args.latent_key].shape[1]
    model = MLP(
        len(hvg_genes),
        args.hidden,
        latent_dim,
        use_layernorm=bool(args.input_layernorm),
        hidden_layernorm=bool(args.hidden_layernorm),
        dropout=float(args.dropout),
        residual_blocks=int(args.residual_blocks),
    )
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)

    module_head: torch.nn.Module | None = None
    module_weight_t: torch.Tensor | None = None
    module_names: list[str] = []
    module_levels: list[str] = []
    module_gene_counts: list[int] = []
    if float(args.lambda_module_loss) > 0:
        if not args.module_target_markers_csv:
            raise ValueError(
                "--lambda-module-loss > 0 requires --module-target-markers-csv."
            )
        module_W, module_names, module_levels, module_gene_counts = (
            build_module_target_matrix(
                module_csv=str(args.module_target_markers_csv),
                hvg_genes=hvg_genes,
                min_genes_per_module=int(args.module_min_genes),
            )
        )
        module_weight_t = torch.from_numpy(module_W).to(device)
        module_head = torch.nn.Linear(latent_dim, module_W.shape[1], bias=True).to(device)
        if module_weight_t.shape[0] != len(hvg_genes):
            raise RuntimeError("module weight matrix shape mismatch with HVG dimension.")
        print(
            f"Enabled module aux loss: n_modules={module_W.shape[1]}, "
            f"min_genes={int(args.module_min_genes)}, lambda={float(args.lambda_module_loss):.4f}"
        )

    domain_classifier: DomainClassifier | None = None
    domain_dann_hidden = parse_int_list(args.domain_dann_hidden)
    if domain_method == "dann":
        if not domain_dann_hidden:
            domain_dann_hidden = [256, 128]
        domain_classifier = DomainClassifier(
            in_dim=latent_dim,
            hidden=domain_dann_hidden,
            dropout=float(args.domain_dann_dropout),
        ).to(device)

    param_groups = get_param_groups(model, float(args.weight_decay))
    if module_head is not None:
        param_groups.extend(get_param_groups(module_head, float(args.weight_decay)))
    if domain_classifier is not None:
        param_groups.extend(get_param_groups(domain_classifier, float(args.weight_decay)))
    opt = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    latent_mean = None
    latent_std = None
    if args.latent_scale == "zscore":
        latent_mean, latent_std = compute_latent_stats(
            adata_ref_train, train_indices, args.latent_key, args.batch_size, rng
        )
        latent_mean_t = torch.from_numpy(latent_mean).to(device)
        latent_std_t = torch.from_numpy(latent_std).to(device)

    coarse_labels_ref = None
    if args.coarse_loss_weight > 0:
        if args.coarse_label_col in adata_ref_train.obs:
            coarse_labels_ref = adata_ref_train.obs[
                args.coarse_label_col
            ].to_numpy()
        else:
            print(f"Missing coarse label column: {args.coarse_label_col}")
            args.coarse_loss_weight = 0.0

    do_log1p = bool(args.input_log1p)
    input_normalize = bool(args.input_normalize)
    input_target_sum = float(args.input_target_sum)
    input_scale = bool(args.input_scale)
    input_scale_mean = None
    input_scale_std = None
    libsize_cache = None
    hvg_cache = None
    if args.cache_libsize:
        print("Caching library size for training reference...")
        libsize_cache = compute_library_size(
            adata_ref_train, train_ref_indices, args.layer_key, args.batch_size
        )
    if args.cache_hvg:
        print("Caching HVG matrix for training reference...")
        hvg_cache = build_hvg_cache(
            adata_ref_train, train_ref_indices, hvg_idx_train, args.layer_key, args.batch_size
        )
    if input_scale:
        input_scale_mean, input_scale_std = compute_scale_stats(
            adata_ref_train,
            train_pool,
            hvg_idx_train,
            args.layer_key,
            do_log1p,
            input_normalize,
            input_target_sum,
            args.batch_size,
            rng,
            hvg_cache=hvg_cache,
            libsize_cache=libsize_cache,
        )

    def _fetch_query_batch_hvg(q_idx: np.ndarray) -> np.ndarray:
        if not missing_mask.any():
            return fetch_X_batch(
                adata_query,
                q_idx,
                present_query_idx,
                args.layer_key,
                do_log1p,
                input_normalize,
                input_target_sum,
                input_scale_mean,
                input_scale_std,
                mask_hvg,
            )
        return fetch_X_batch_aligned(
            adata_query,
            q_idx,
            len(hvg_genes),
            args.layer_key,
            do_log1p,
            input_normalize,
            input_target_sum,
            input_scale_mean,
            input_scale_std,
            mask_hvg,
            present_query_idx,
            hvg_present_pos,
            query_full,
            full_layer_key,
            present_full_idx,
            hvg_full_pos,
            missing_pos,
        )

    best_val = float("inf")
    best_epoch = 0
    patience = max(0, int(args.early_stop_patience))
    min_delta = float(args.early_stop_min_delta)
    no_improve = 0
    best_path = os.path.join(args.out_root, "model", "best_encoder.pt")
    os.makedirs(os.path.join(args.out_root, "model"), exist_ok=True)
    steps_per_epoch = max(1, math.ceil(len(train_indices) / max(1, args.batch_size)))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(round(float(args.warmup_frac) * total_steps))
    min_lr = float(args.lr) * float(args.min_lr_factor)
    global_step = 0

    def _set_lr(step: int) -> None:
        if total_steps <= 0:
            return
        if warmup_steps > 0 and step < warmup_steps:
            lr = float(args.lr) * float(step + 1) / float(warmup_steps)
        else:
            denom = max(1, total_steps - warmup_steps)
            t = float(step - warmup_steps) / float(denom)
            t = min(max(t, 0.0), 1.0)
            lr = min_lr + 0.5 * (float(args.lr) - min_lr) * (1.0 + math.cos(math.pi * t))
        for pg in opt.param_groups:
            pg["lr"] = lr

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        if module_head is not None:
            module_head.train()
        if domain_classifier is not None:
            domain_classifier.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_coarse = 0.0
        epoch_cos = 0.0
        epoch_domain = 0.0
        epoch_module = 0.0
        n_seen = 0
        for batch_idx in iter_batches(train_indices, args.batch_size, rng, True):
            _set_lr(global_step)
            X = fetch_X_batch(
                adata_ref_train,
                batch_idx,
                hvg_idx_train,
                args.layer_key,
                do_log1p,
                input_normalize,
                input_target_sum,
                input_scale_mean,
                input_scale_std,
                mask_hvg,
                hvg_cache=hvg_cache,
                libsize_cache=libsize_cache,
            )
            Z = fetch_Z_batch(adata_ref_train, batch_idx, args.latent_key)
            x_raw_t = torch.from_numpy(X).to(device)
            x_t = x_raw_t
            if args.gene_dropout and float(args.gene_dropout) > 0:
                x_t = F.dropout(x_t, p=float(args.gene_dropout), training=True)
            z_t = torch.from_numpy(Z).to(device)
            opt.zero_grad()
            pred = model(x_t)
            if args.latent_scale == "zscore":
                pred_s = (pred - latent_mean_t) / latent_std_t
                z_s = (z_t - latent_mean_t) / latent_std_t
            else:
                pred_s = pred
                z_s = z_t

            if args.mse_by_label and coarse_labels_ref is not None:
                coarse_batch = coarse_labels_ref[batch_idx]
                recon_loss = mse_within_labels(pred_s, z_s, coarse_batch)
            else:
                recon_loss = loss_fn(pred_s, z_s)

            coarse_loss = torch.tensor(0.0, device=device)
            if args.coarse_loss_weight > 0 and coarse_labels_ref is not None:
                coarse_batch = coarse_labels_ref[batch_idx]
                coarse_loss = coarse_contrastive_loss(
                    pred_s,
                    coarse_batch,
                    args.coarse_margin,
                    args.coarse_subsample,
                    rng,
                )
            cos_loss = torch.tensor(0.0, device=device)
            if args.cosine_loss_weight and float(args.cosine_loss_weight) > 0:
                cos_sim = F.cosine_similarity(pred_s, z_s, dim=1)
                cos_loss = (1.0 - cos_sim).mean()
            domain_loss = torch.tensor(0.0, device=device)
            if domain_method != "none":
                replace = domain_batch_size > domain_query_pool.size
                q_idx = rng.choice(
                    domain_query_pool, size=domain_batch_size, replace=replace
                )
                Xq = _fetch_query_batch_hvg(q_idx)
                xq_t = torch.from_numpy(Xq).to(device)
                if args.gene_dropout and float(args.gene_dropout) > 0:
                    xq_t = F.dropout(xq_t, p=float(args.gene_dropout), training=True)
                pred_q = model(xq_t)
                if args.latent_scale == "zscore":
                    pred_q_s = (pred_q - latent_mean_t) / latent_std_t
                else:
                    pred_q_s = pred_q
                if domain_method == "coral":
                    domain_loss = coral_alignment_loss(
                        pred_s,
                        pred_q_s,
                        mean_weight=float(args.domain_coral_mean_weight),
                    )
                elif domain_method == "mmd":
                    domain_loss = mmd_alignment_loss(
                        pred_s,
                        pred_q_s,
                        kernel=str(args.domain_mmd_kernel),
                        gamma=float(args.domain_mmd_gamma),
                    )
                elif domain_method == "dann":
                    if domain_classifier is None:
                        raise RuntimeError("Domain classifier is missing for DANN mode.")
                    feat = torch.cat(
                        [
                            grad_reverse(pred_s, float(args.domain_dann_grl_coeff)),
                            grad_reverse(pred_q_s, float(args.domain_dann_grl_coeff)),
                        ],
                        dim=0,
                    )
                    y_dom = torch.cat(
                        [
                            torch.zeros(pred_s.shape[0], dtype=torch.long, device=device),
                            torch.ones(pred_q_s.shape[0], dtype=torch.long, device=device),
                        ],
                        dim=0,
                    )
                    logits_dom = domain_classifier(feat)
                    domain_loss = F.cross_entropy(logits_dom, y_dom)
                else:
                    raise RuntimeError(f"Unknown domain method: {domain_method}")
            module_loss = torch.tensor(0.0, device=device)
            if module_head is not None and module_weight_t is not None:
                module_target = x_raw_t @ module_weight_t
                module_pred = module_head(pred)
                module_loss = loss_fn(module_pred, module_target)
            loss = (
                recon_loss
                + args.coarse_loss_weight * coarse_loss
                + float(args.cosine_loss_weight) * cos_loss
                + float(domain_weight) * domain_loss
                + float(args.lambda_module_loss) * module_loss
            )
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch_idx)
            epoch_recon += recon_loss.item() * len(batch_idx)
            epoch_coarse += coarse_loss.item() * len(batch_idx)
            epoch_cos += cos_loss.item() * len(batch_idx)
            epoch_domain += domain_loss.item() * len(batch_idx)
            epoch_module += module_loss.item() * len(batch_idx)
            n_seen += len(batch_idx)
            global_step += 1
        epoch_loss /= max(1, n_seen)
        epoch_recon /= max(1, n_seen)
        epoch_coarse /= max(1, n_seen)
        epoch_cos /= max(1, n_seen)
        epoch_domain /= max(1, n_seen)
        epoch_module /= max(1, n_seen)
        epoch_coarse_weighted = epoch_coarse * args.coarse_loss_weight

        model.eval()
        if module_head is not None:
            module_head.eval()
        if domain_classifier is not None:
            domain_classifier.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_cos = 0.0
        val_module = 0.0
        n_val_seen = 0
        val_adata = adata_ref_val if adata_ref_val is not None else adata_ref_train
        val_hvg_idx = hvg_idx_val if adata_ref_val is not None else hvg_idx_train
        val_hvg_cache = None
        val_libsize_cache = None
        with torch.no_grad():
            for batch_idx in iter_batches(val_indices, args.batch_size, rng, False):
                X = fetch_X_batch(
                    val_adata,
                    batch_idx,
                    val_hvg_idx,
                    args.layer_key,
                    do_log1p,
                    input_normalize,
                    input_target_sum,
                    input_scale_mean,
                    input_scale_std,
                    mask_hvg,
                    hvg_cache=val_hvg_cache,
                    libsize_cache=val_libsize_cache,
                )
                Z = fetch_Z_batch(val_adata, batch_idx, args.latent_key)
                x_raw_t = torch.from_numpy(X).to(device)
                x_t = x_raw_t
                z_t = torch.from_numpy(Z).to(device)
                pred = model(x_t)
                if args.latent_scale == "zscore":
                    pred_s = (pred - latent_mean_t) / latent_std_t
                    z_s = (z_t - latent_mean_t) / latent_std_t
                else:
                    pred_s = pred
                    z_s = z_t
                recon_loss = loss_fn(pred_s, z_s)
                cos_loss = torch.tensor(0.0, device=device)
                if args.cosine_loss_weight and float(args.cosine_loss_weight) > 0:
                    cos_sim = F.cosine_similarity(pred_s, z_s, dim=1)
                    cos_loss = (1.0 - cos_sim).mean()
                module_loss = torch.tensor(0.0, device=device)
                if module_head is not None and module_weight_t is not None:
                    module_target = x_raw_t @ module_weight_t
                    module_pred = module_head(pred)
                    module_loss = loss_fn(module_pred, module_target)
                loss = (
                    recon_loss
                    + float(args.cosine_loss_weight) * cos_loss
                    + float(args.lambda_module_loss) * module_loss
                )
                val_loss += loss.item() * len(batch_idx)
                val_recon += recon_loss.item() * len(batch_idx)
                val_cos += cos_loss.item() * len(batch_idx)
                val_module += module_loss.item() * len(batch_idx)
                n_val_seen += len(batch_idx)
        val_loss /= max(1, n_val_seen)
        val_recon /= max(1, n_val_seen)
        val_cos /= max(1, n_val_seen)
        val_module /= max(1, n_val_seen)
        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_recon": epoch_recon,
            "train_coarse": epoch_coarse,
            "train_cosine": epoch_cos,
            "train_domain": epoch_domain,
            "train_module": epoch_module,
            "lambda_coarse": float(args.coarse_loss_weight),
            "domain_method": str(domain_method),
            "lambda_domain": float(domain_weight),
            "lambda_domain_coral": float(args.domain_coral_weight),
            "lambda_domain_dann": float(args.domain_dann_weight),
            "lambda_domain_mmd": float(args.domain_mmd_weight),
            "lambda_module": float(args.lambda_module_loss),
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_cosine": val_cos,
            "val_module": val_module,
        }
        history.append(epoch_record)

        print(
            "epoch {e}: train_loss={tl:.6f} recon={rl:.6f} coarse={cl:.6f} "
            "cosine={cs:.6f} domain={td:.6f} module={tm:.6f} "
            "lambda_coarse={lc:.6f} domain_method={dm} lambda_domain={ld:.6f} lambda_module={lm:.6f} "
            "val_loss={vl:.6f} val_recon={vr:.6f} val_cosine={vc:.6f} val_module={vm:.6f}".format(
                e=epoch,
                tl=epoch_loss,
                rl=epoch_recon,
                cl=epoch_coarse,
                cs=epoch_cos,
                td=epoch_domain,
                tm=epoch_module,
                lc=epoch_coarse_weighted,
                dm=str(domain_method),
                ld=float(domain_weight),
                lm=float(args.lambda_module_loss),
                vl=val_loss,
                vr=val_recon,
                vc=val_cos,
                vm=val_module,
            )
        )
        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hvg_idx": hvg_idx,
                    "hvg_genes": hvg_genes,
                    "latent_dim": latent_dim,
                    "latent_scale": args.latent_scale,
                    "latent_mean": latent_mean,
                    "latent_std": latent_std,
                    "input_log1p": bool(do_log1p),
                    "input_normalize": input_normalize,
                    "input_target_sum": input_target_sum,
                    "input_scale": input_scale,
                    "input_scale_mean": input_scale_mean,
                    "input_scale_std": input_scale_std,
                    "input_layernorm": bool(args.input_layernorm),
                    "use_layernorm": bool(args.input_layernorm),
                    "hidden_layernorm": bool(args.hidden_layernorm),
                    "residual_blocks": int(args.residual_blocks),
                    "coarse_label_col": args.coarse_label_col,
                    "mse_by_label": args.mse_by_label,
                    "module_aux_enabled": bool(module_head is not None),
                    "module_target_markers_csv": args.module_target_markers_csv,
                    "module_min_genes": int(args.module_min_genes),
                    "lambda_module_loss": float(args.lambda_module_loss),
                    "domain_method": str(domain_method),
                    "lambda_domain": float(domain_weight),
                    "lambda_domain_coral": float(args.domain_coral_weight),
                    "domain_coral_mean_weight": float(args.domain_coral_mean_weight),
                    "lambda_domain_dann": float(args.domain_dann_weight),
                    "domain_dann_grl_coeff": float(args.domain_dann_grl_coeff),
                    "domain_dann_hidden": domain_dann_hidden,
                    "domain_dann_dropout": float(args.domain_dann_dropout),
                    "lambda_domain_mmd": float(args.domain_mmd_weight),
                    "domain_mmd_kernel": str(args.domain_mmd_kernel),
                    "domain_mmd_gamma": float(args.domain_mmd_gamma),
                    "domain_batch_size": int(domain_batch_size),
                    "module_names": module_names,
                    "module_levels": module_levels,
                    "module_gene_counts": module_gene_counts,
                    "module_head_state_dict": (
                        module_head.state_dict() if module_head is not None else None
                    ),
                    "domain_classifier_state_dict": (
                        domain_classifier.state_dict()
                        if domain_classifier is not None
                        else None
                    ),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                best_path,
            )
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_val={best_val:.6f})."
                )
                break

    model_path = os.path.join(args.out_root, "model", "encoder.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hvg_idx": hvg_idx,
            "hvg_genes": hvg_genes,
            "latent_dim": latent_dim,
            "latent_scale": args.latent_scale,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "input_log1p": bool(do_log1p),
            "input_normalize": input_normalize,
            "input_target_sum": input_target_sum,
            "input_scale": input_scale,
            "input_scale_mean": input_scale_mean,
            "input_scale_std": input_scale_std,
            "input_layernorm": bool(args.input_layernorm),
            "use_layernorm": bool(args.input_layernorm),
            "hidden_layernorm": bool(args.hidden_layernorm),
            "residual_blocks": int(args.residual_blocks),
            "coarse_label_col": args.coarse_label_col,
            "mse_by_label": args.mse_by_label,
            "module_aux_enabled": bool(module_head is not None),
            "module_target_markers_csv": args.module_target_markers_csv,
            "module_min_genes": int(args.module_min_genes),
            "lambda_module_loss": float(args.lambda_module_loss),
            "domain_method": str(domain_method),
            "lambda_domain": float(domain_weight),
            "lambda_domain_coral": float(args.domain_coral_weight),
            "domain_coral_mean_weight": float(args.domain_coral_mean_weight),
            "lambda_domain_dann": float(args.domain_dann_weight),
            "domain_dann_grl_coeff": float(args.domain_dann_grl_coeff),
            "domain_dann_hidden": domain_dann_hidden,
            "domain_dann_dropout": float(args.domain_dann_dropout),
            "lambda_domain_mmd": float(args.domain_mmd_weight),
            "domain_mmd_kernel": str(args.domain_mmd_kernel),
            "domain_mmd_gamma": float(args.domain_mmd_gamma),
            "domain_batch_size": int(domain_batch_size),
            "module_names": module_names,
            "module_levels": module_levels,
            "module_gene_counts": module_gene_counts,
            "module_head_state_dict": (
                module_head.state_dict() if module_head is not None else None
            ),
            "domain_classifier_state_dict": (
                domain_classifier.state_dict()
                if domain_classifier is not None
                else None
            ),
        },
        model_path,
    )
    with open(os.path.join(args.out_root, "model", "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.out_root, "model", "loss_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    if module_head is not None:
        with open(os.path.join(args.out_root, "model", "module_aux_config.json"), "w") as f:
            json.dump(
                {
                    "module_target_markers_csv": args.module_target_markers_csv,
                    "module_min_genes": int(args.module_min_genes),
                    "lambda_module_loss": float(args.lambda_module_loss),
                    "n_modules": int(len(module_names)),
                    "module_names": module_names,
                    "module_levels": module_levels,
                    "module_gene_counts": module_gene_counts,
                },
                f,
                indent=2,
            )
    if os.path.exists(best_path):
        with open(os.path.join(args.out_root, "model", "best_checkpoint.json"), "w") as f:
            json.dump({"best_epoch": best_epoch, "best_val_loss": best_val}, f, indent=2)

    # Projection runtime on query
    model.eval()
    query_n = adata_query.n_obs
    query_latent = np.zeros((query_n, latent_dim), dtype=np.float32)
    start = time.perf_counter()
    with torch.no_grad():
        for i, batch_idx in enumerate(
            iter_batches(np.arange(query_n), args.batch_size, rng, False)
        ):
            X = _fetch_query_batch_hvg(batch_idx)
            x_t = torch.from_numpy(X).to(device)
            pred = model(x_t).cpu().numpy()
            query_latent[batch_idx] = pred
            if (i + 1) % 50 == 0:
                print(f"projected {int((i + 1) * args.batch_size)} cells")
    elapsed = time.perf_counter() - start
    throughput = query_n / max(1e-6, elapsed)

    runtime_path = os.path.join(args.out_root, "runtime", "encoder_projection.json")
    with open(runtime_path, "w") as f:
        json.dump(
            {"n_cells": int(query_n), "seconds": elapsed, "cells_per_sec": throughput},
            f,
            indent=2,
        )

    np.save(os.path.join(args.out_root, "projection", "encoder_latent.npy"), query_latent)
    pd.DataFrame(
        query_latent,
        index=adata_query.obs_names,
    ).to_csv(os.path.join(args.out_root, "projection", "encoder_latent.csv"))

    # Label transfer
    ref_latent = fetch_Z_batch(adata_ref_full, ref_indices, args.latent_key)
    ref_labels = adata_ref_full.obs[args.label_col].astype(str).to_numpy()[ref_indices]
    query_labels = adata_query.obs[args.label_col].astype(str).to_numpy()
    assigned_mask = ref_labels != "unassigned"
    ref_latent_knn = ref_latent[assigned_mask]
    ref_labels_knn = ref_labels[assigned_mask]
    ref_tissue_knn = None
    query_tissue = None
    if args.tissue_conditioned and tissue_col:
        ref_tissue = adata_ref_full.obs[tissue_col].astype(str).to_numpy()[ref_indices]
        ref_tissue_knn = ref_tissue[assigned_mask]
        query_tissue = adata_query.obs[tissue_col].astype(str).to_numpy()
    ref_latent_knn_use = ref_latent_knn
    query_latent_use = query_latent
    if args.knn_l2norm:
        ref_latent_knn_use = l2_normalize(ref_latent_knn_use)
        query_latent_use = l2_normalize(query_latent_use)
    knn_idx, knn_dist = tissue_conditioned_knn_neighbors(
        ref_latent_knn_use,
        ref_tissue_knn,
        query_latent_use,
        query_tissue,
        args.k,
        args.tissue_aware_mode if args.tissue_conditioned else "none",
        args.tissue_penalty_lambda,
        args.knn_metric,
    )
    pred_labels, max_prob, entropy, mean_dist = knn_labels_from_neighbors(
        ref_labels_knn, knn_idx, knn_dist
    )

    entropy_p95 = float(np.quantile(entropy, 0.95))
    low_conf = max_prob < args.confidence_threshold
    if args.confidence_threshold > 0:
        pred_labels = pred_labels.astype(object)
        pred_labels[low_conf] = "unassigned"
    high_entropy = entropy >= entropy_p95
    pred_df = pd.DataFrame(
        {
            "pred_label": pred_labels,
            "true_label": query_labels,
            "knn_max_prob": max_prob,
            "knn_entropy": entropy,
            "knn_mean_dist": mean_dist,
            "low_confidence": low_conf,
            "high_entropy": high_entropy,
        },
        index=adata_query.obs_names,
    )
    pred_df.to_csv(
        os.path.join(args.out_root, "label_transfer", "encoder_pred_labels.csv")
    )

    label_list = sorted(np.unique(ref_labels_knn).tolist())
    if args.confidence_threshold > 0 and "unassigned" not in label_list:
        label_list.append("unassigned")
    exclude_unassigned = not args.include_unassigned
    omit_eval_labels = parse_label_list(args.omit_eval_labels)
    if exclude_unassigned:
        label_list = [lbl for lbl in label_list if lbl != "unassigned"]
        mask = (query_labels != "unassigned") & ~pd.isna(query_labels)
        save_metrics(
            query_labels[mask],
            pred_labels[mask],
            label_list,
            os.path.join(args.out_root, "metrics"),
            omit_true_labels=omit_eval_labels,
        )
    else:
        save_metrics(
            query_labels,
            pred_labels,
            label_list,
            os.path.join(args.out_root, "metrics"),
            omit_true_labels=omit_eval_labels,
        )

    conf_summary = {
        "entropy_p50": float(np.quantile(entropy, 0.5)),
        "entropy_p90": float(np.quantile(entropy, 0.9)),
        "entropy_p95": entropy_p95,
        "max_prob_p10": float(np.quantile(max_prob, 0.1)),
        "max_prob_p05": float(np.quantile(max_prob, 0.05)),
        "mean_dist_p50": float(np.quantile(mean_dist, 0.5)),
        "mean_dist_p90": float(np.quantile(mean_dist, 0.9)),
        "mean_dist_p95": float(np.quantile(mean_dist, 0.95)),
        "low_confidence_frac": float(low_conf.mean()),
        "high_entropy_frac": float(high_entropy.mean()),
    }
    with open(
        os.path.join(args.out_root, "label_transfer", "encoder_confidence_summary.json"),
        "w",
    ) as f:
        json.dump(conf_summary, f, indent=2)

    umap_keep_mask = None
    umap_filter_quantile = args.umap_filter_quantile
    if args.latent_filter_quantile > 0:
        umap_keep_mask = latent_label_distance_mask(
            ref_latent,
            ref_labels,
            query_latent,
            args.latent_filter_label,
            args.latent_filter_metric,
            args.latent_filter_quantile,
            args.latent_filter_k,
        )
        if umap_keep_mask is not None and umap_filter_quantile > 0:
            print("UMAP distance filter disabled because latent filter is enabled.")
            umap_filter_quantile = 0.0

    # Time regression (kNN on latent)
    if args.time_col:
        if args.time_col not in adata_ref_full.obs:
            print(
                f"Time column {args.time_col} not found in reference; skipping time regression."
            )
        else:
            ref_time_raw = adata_ref_full.obs[args.time_col].to_numpy()
            ref_valid = ~pd.isna(ref_time_raw)
            ref_time_labels = [
                str(x)
                for x in ref_time_raw[ref_valid]
                if str(x) != "unassigned"
            ]
            ref_categories = list(pd.unique(ref_time_labels))
            if not ref_categories:
                print("No valid reference time values; skipping time regression.")
            else:
                if args.time_order:
                    order = [t.strip() for t in args.time_order.split(",") if t.strip()]
                    missing = set(ref_categories) - set(order)
                    if missing:
                        raise ValueError(
                            f"time_order missing categories: {sorted(missing)}"
                        )
                else:
                    order = _infer_time_order(ref_categories)
                time_map = {label: i for i, label in enumerate(order)}
                mapping_path = os.path.join(
                    args.out_root, "time_regression", "time_mapping.json"
                )
                with open(mapping_path, "w") as f:
                    json.dump(
                        {
                            "time_col": args.time_col,
                            "order": order,
                            "mapping": time_map,
                        },
                        f,
                        indent=2,
                    )

                ref_time_enc = np.full(ref_time_raw.shape[0], np.nan, dtype=np.float32)
                for idx, label in zip(np.flatnonzero(ref_valid), ref_time_labels):
                    ref_time_enc[idx] = float(time_map[label])
                ref_time_enc = ref_time_enc[ref_indices]
                ref_time_label_knn = (
                    adata_ref_full.obs[args.time_col]
                    .astype(str)
                    .to_numpy()[ref_indices][assigned_mask]
                )

                query_time_raw = None
                query_time_enc = None
                if args.time_col in adata_query.obs:
                    query_time_raw = adata_query.obs[args.time_col].to_numpy()
                    query_time_enc = np.full(
                        query_time_raw.shape[0], np.nan, dtype=np.float32
                    )
                    q_valid = ~pd.isna(query_time_raw)
                    for idx, label in zip(
                        np.flatnonzero(q_valid),
                        [str(x) for x in query_time_raw[q_valid]],
                    ):
                        if label != "unassigned" and label in time_map:
                            query_time_enc[idx] = float(time_map[label])

                ref_time_knn = ref_time_enc[assigned_mask]
                if np.isfinite(ref_time_knn).sum() == 0:
                    print("No valid reference time values after filtering; skipping time regression.")
                else:
                    time_pred, time_var, time_mean_dist, time_prob = (
                        time_regression_from_neighbors(
                            ref_time_knn,
                            knn_idx,
                            knn_dist,
                            0.0,
                            len(order),
                            args.time_topk,
                            query_time_enc,
                            args.time_monotone_delta,
                            args.time_monotone_gamma,
                            args.time_trim_extremes,
                        )
                    )
                    time_entropy = np.full(query_latent.shape[0], np.nan, dtype=np.float32)
                    prob_sum = time_prob.sum(axis=1)
                    valid_prob = prob_sum > 0
                    if np.any(valid_prob):
                        p = time_prob[valid_prob] / prob_sum[valid_prob][:, None]
                        time_entropy[valid_prob] = -np.sum(p * np.log(p + 1e-12), axis=1)
                    entropy_p95 = float(np.nanquantile(time_entropy, 0.95))
                    high_time_entropy = time_entropy >= entropy_p95
                    time_pred[high_time_entropy] = np.nan
                    time_var[high_time_entropy] = np.nan
                    time_prob[high_time_entropy] = 0.0
                    delay = np.full(query_latent.shape[0], np.nan, dtype=np.float32)
                    time_pred_label = np.full(query_latent.shape[0], None, dtype=object)
                    time_pred_ordinal = np.full(
                        query_latent.shape[0], np.nan, dtype=np.float32
                    )
                    for i in range(query_latent.shape[0]):
                        row_idx = knn_idx[i]
                        valid = row_idx >= 0
                        if not np.any(valid):
                            continue
                        row_idx = row_idx[valid]
                        row_dist = knn_dist[i][valid]
                        row_ord = ref_time_knn[row_idx]
                        valid_time = np.isfinite(row_ord)
                        if not np.any(valid_time):
                            continue
                        row_idx = row_idx[valid_time]
                        row_dist = row_dist[valid_time]
                        row_labels = ref_time_label_knn[row_idx]
                        if args.time_hard_topk > 0:
                            keep = min(args.time_hard_topk, row_idx.size)
                            order_idx = np.argsort(row_dist)[:keep]
                            row_idx = row_idx[order_idx]
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
                            key=lambda x: (
                                -counts[x],
                                dist_sums.get(x, 0.0) / counts[x],
                                time_map.get(x, 0),
                            ),
                        )[0]
                        time_pred_label[i] = best
                        time_pred_ordinal[i] = float(time_map[best])

                    if query_time_enc is not None:
                        valid_time = np.isfinite(query_time_enc)
                        delay[valid_time] = (
                            time_pred[valid_time] - query_time_enc[valid_time]
                        )
                        valid_acc = valid_time & np.isfinite(time_pred_ordinal)
                        if np.any(valid_acc):
                            y_true = np.array(
                                [str(x) for x in query_time_raw[valid_acc]]
                            )
                            y_pred = time_pred_label[valid_acc].astype(str)
                            diff = np.abs(
                                time_pred_ordinal[valid_acc]
                                - query_time_enc[valid_acc]
                            )
                            exact_acc = float(np.mean(diff == 0))
                            ordinal_acc = float(np.mean(diff <= 1))
                            mae = float(np.mean(diff))
                            with open(
                                os.path.join(
                                    args.out_root,
                                    "time_regression",
                                    "time_accuracy.txt",
                                ),
                                "w",
                            ) as f:
                                f.write(
                                    f"n_valid={int(valid_acc.sum())}\n"
                                    f"exact_accuracy={exact_acc:.6f}\n"
                                    f"ordinal_accuracy_±1={ordinal_acc:.6f}\n"
                                    f"mean_abs_error={mae:.6f}\n"
                                )
                            cm = confusion_matrix(y_true, y_pred, labels=order)
                            cm_df = pd.DataFrame(cm, index=order, columns=order)
                            cm_df.to_csv(
                                os.path.join(
                                    args.out_root,
                                    "time_regression",
                                    "time_confusion_matrix.csv",
                                )
                            )
                            plot_confusion_matrix(
                                cm,
                                order,
                                os.path.join(args.out_root, "time_regression"),
                                "_time",
                                normalize=False,
                            )

                    ref_time_df = pd.DataFrame(
                        {
                            "ref_cell_id": ref_obs_names[ref_indices][assigned_mask],
                            "time_label": adata_ref_full.obs[args.time_col]
                            .astype(str)
                            .to_numpy()[ref_indices][assigned_mask],
                            "time_ordinal": ref_time_knn,
                        }
                    )
                    if ref_tissue_knn is not None:
                        ref_time_df["ref_tissue"] = ref_tissue_knn
                    ref_time_df.to_csv(
                        os.path.join(
                            args.out_root, "time_regression", "ref_time_mapping.csv"
                        ),
                        index=False,
                    )

                    query_time_df = pd.DataFrame(
                        {
                            "time_pred": time_pred,
                            "time_var": time_var,
                            "time_delay": delay,
                            "time_obs": query_time_raw,
                            "time_obs_ordinal": query_time_enc,
                            "time_pred_label": time_pred_label,
                            "time_pred_ordinal": time_pred_ordinal,
                            "knn_mean_dist": time_mean_dist,
                        },
                        index=adata_query.obs_names,
                    )
                    query_time_df.to_csv(
                        os.path.join(
                            args.out_root,
                            "time_regression",
                            "query_time_predictions.csv",
                        )
                    )
                    prob_df = pd.DataFrame(
                        time_prob, columns=order, index=adata_query.obs_names
                    )
                    prob_df.to_csv(
                        os.path.join(
                            args.out_root,
                            "time_regression",
                            "time_probabilities.csv",
                        )
                    )

    # Hierarchy-aware evaluation
    accuracy_by_level = {}
    level_cols = {
        "ZMAP_GermLayer": "ZMAP_GermLayer",
        "ZMAP_CellType": args.label_col,
        "ZMAP_CellTypeFine": "ZMAP_CellTypeFine",
    }
    if tissue_col:
        level_cols["ZMAP_tissue"] = tissue_col
    for level_name, col in level_cols.items():
        if col not in adata_ref_full.obs or col not in adata_query.obs:
            continue
        ref_lab = adata_ref_full.obs[col].astype(str).to_numpy()[ref_indices]
        qry_lab = adata_query.obs[col].astype(str).to_numpy()
        exclude = exclude_unassigned and level_name in (
            "ZMAP_CellType",
            "ZMAP_CellTypeFine",
            "ZMAP_tissue",
            "ZMAP_GermLayer",
        )
        acc = evaluate_label_level(
            level_name,
            ref_latent_knn,
            ref_lab[assigned_mask],
            query_latent,
            qry_lab,
            args.k,
            os.path.join(args.out_root, "metrics"),
            exclude,
            args.confidence_threshold,
            ref_tissue_knn,
            query_tissue,
            args.tissue_aware_mode if args.tissue_conditioned else "none",
            args.tissue_penalty_lambda,
            omit_eval_labels,
        )
        accuracy_by_level[level_name] = float(acc)
    with open(
        os.path.join(args.out_root, "metrics", "accuracy_by_level.json"), "w"
    ) as f:
        json.dump(accuracy_by_level, f, indent=2)

    if not args.no_umap:
        umap_ref_latent = fetch_Z_batch(
            adata_umap_ref,
            np.arange(adata_umap_ref.n_obs),
            args.latent_key,
        )
        ref_umap = None
        ref_umap_labels = None
        label_palette = None
        if args.umap_use_precomputed:
            ref_umap = np.asarray(adata_umap_ref.obsm["X_umap"])
            if args.label_col in adata_umap_ref.obs:
                ref_umap_labels = adata_umap_ref.obs[args.label_col].to_numpy()
                if args.label_col == "ZMAP_CellType":
                    if "ZMAP_colormap_C79" not in adata_umap_ref.uns:
                        raise ValueError("Missing adata.uns['ZMAP_colormap_C79'] for ZMAP_CellType.")
                    raw_palette = dict(adata_umap_ref.uns["ZMAP_colormap_C79"])
                    label_palette = {str(k): _as_str(v) for k, v in raw_palette.items()}
                else:
                    try:
                        cats = adata_umap_ref.obs[args.label_col].astype("category").cat.categories
                        color_key = f"{args.label_col}_colors"
                        cmap_key = f"{args.label_col}_color_map"
                        if cmap_key in adata_umap_ref.uns:
                            label_palette = dict(adata_umap_ref.uns[cmap_key])
                        elif color_key in adata_umap_ref.uns:
                            cols = adata_umap_ref.uns[color_key]
                            if len(cols) >= len(cats):
                                label_palette = dict(zip(cats, cols))
                    except Exception:
                        label_palette = None
        umap_knn_k = args.umap_knn_k if args.umap_knn_k > 0 else args.k
        umap_proj_labels = None
        umap_knn_idx = None
        umap_knn_dist = None
        umap_knn_ref_latent = None
        umap_knn_ref_labels = None
        umap_knn_ref_ids = None
        if args.umap_proj_label_filter:
            umap_proj_labels = pred_labels
            if umap_knn_k != args.k:
                print("UMAP label-proj: umap_knn_k differs from k; recomputing kNN.")
            else:
                umap_knn_idx = knn_idx
                umap_knn_dist = knn_dist
                umap_knn_ref_latent = ref_latent_knn_use
                umap_knn_ref_labels = ref_labels_knn
                umap_knn_ref_ids = ref_obs_names.to_numpy()[ref_indices][assigned_mask]
        query_coords = maybe_umap_overlay(
            umap_ref_latent,
            query_latent,
            query_labels,
            adata_query.obs_names.astype(str).to_numpy(),
            os.path.join(args.out_root, "umap"),
            args.seed,
            args.umap_max_ref,
            args.label_col,
            args.umap_use_precomputed,
            ref_umap,
            adata_umap_ref,
            umap_knn_k,
            args.umap_proj_mode,
            args.umap_medoid_topm,
            args.umap_medoid_jitter,
            args.umap_plot_jitter,
            args.umap_query_max_dist,
            args.umap_query_dist_quantile,
            ref_umap_labels,
            args.umap_weight,
            args.umap_tau,
            args.umap_tau_quantile,
            umap_filter_quantile,
            umap_keep_mask,
            label_palette,
            None,
            None,
            umap_proj_label_filter=args.umap_proj_label_filter,
            umap_proj_labels=umap_proj_labels,
            umap_knn_idx=umap_knn_idx,
            umap_knn_dist=umap_knn_dist,
            umap_knn_ref_latent=umap_knn_ref_latent,
            umap_knn_ref_labels=umap_knn_ref_labels,
            umap_knn_ref_ids=umap_knn_ref_ids,
        )
        if query_coords is not None:
            adata_query.obsm["X_umap_proj"] = query_coords


if __name__ == "__main__":
    main()
