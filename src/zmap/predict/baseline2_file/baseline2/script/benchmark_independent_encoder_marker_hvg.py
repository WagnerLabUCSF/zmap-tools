#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import anndata as ad
import h5py
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Local training helpers (MLP train/infer).
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_celltype_torch_mlp_lgbxt_ensemble as base  # noqa: E402

# Encoder model + preprocessing helpers.
ENCODER_SCRIPT_DIR = SCRIPT_DIR.parent.parent / "scripts"
sys.path.insert(0, str(ENCODER_SCRIPT_DIR))
import train_encoder_ensembly as encmod  # noqa: E402


@dataclass
class SplitFeatures:
    x_train: np.ndarray
    x_val: np.ndarray
    x_query: np.ndarray
    train_obs: pd.Index
    val_obs: pd.Index
    query_obs: pd.Index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark on independent dataset: "
            "Line1=HVG encoder latent + marker -> MLP (baseline/CORAL/MMD/DANN), "
            "Line2=HVG+marker -> LightGBM."
        )
    )
    p.add_argument(
        "--train-h5ad",
        type=Path,
        default=Path("not_bundled/ref_train_with_reftrainfit_pca.h5ad"),  #change path
    )
    p.add_argument(
        "--val-h5ad",
        type=Path,
        default=Path("not_bundled/ref_val_with_reftrainfit_pca.h5ad"),  #change path
    )
    p.add_argument(
        "--query-h5ad",
        type=Path,
        default=Path("not_bundled/query_with_reftrainfit_pca.h5ad"),  #change path
    )
    p.add_argument("--label-col", type=str, default="ZMAP_CellType")
    p.add_argument("--exclude-labels", type=str, default="unassigned,blastomere,blastomeres")

    p.add_argument(
        "--hvg-csv",
        type=Path,
        default=Path("not_bundled/hvg_from_ref_for_compare.csv"),  #change path
    )
    p.add_argument("--hvg-gene-col", type=str, default="gene")
    p.add_argument(
        "--marker-csv",
        type=Path,
        default=Path("not_bundled/ZMAP_CellType_selected_markers.csv"),  #change path
    )
    p.add_argument("--marker-gene-col", type=str, default="gene")

    p.add_argument("--count-layer", type=str, default="raw_nolog")
    p.add_argument("--target-sum", type=float, default=1e6)
    p.add_argument("--zscore-eps", type=float, default=1e-6)

    p.add_argument(
        "--encoder-root",
        type=Path,
        default=Path("not_bundled/encoder_independent_hvg_ablation"),  #change path
        help="Root containing encoder dirs: hvg_only_baseline/coral/mmd/dann",
    )
    p.add_argument("--encoder-methods", type=str, default="baseline,coral,mmd,dann")
    p.add_argument("--use-best-checkpoint", action="store_true")
    p.add_argument("--projection-batch-size", type=int, default=8192)

    p.add_argument("--mlp-hidden-dims", type=str, default="256,128")
    p.add_argument("--mlp-dropout", type=float, default=0.1)
    p.add_argument("--mlp-lr", type=float, default=0.01)
    p.add_argument("--mlp-batch-size", type=int, default=1024)
    p.add_argument("--mlp-max-epochs", type=int, default=200)
    p.add_argument("--mlp-label-smoothing", type=float, default=0.0)
    p.add_argument("--mlp-patience", type=int, default=20)
    p.add_argument("--mlp-min-delta", type=float, default=1e-4)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-5)

    p.add_argument("--lgb-learning-rate", type=float, default=0.05)
    p.add_argument("--lgb-n-estimators", type=int, default=5000)
    p.add_argument("--lgb-num-leaves", type=int, default=255)
    p.add_argument("--lgb-min-child-samples", type=int, default=20)
    p.add_argument("--lgb-colsample-bytree", type=float, default=1.0)
    p.add_argument("--lgb-subsample", type=float, default=1.0)
    p.add_argument("--lgb-early-stopping-rounds", type=int, default=100)
    p.add_argument("--lgb-n-jobs", type=int, default=-1)

    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--threshold-grid", type=str, default="0.5,0.6,0.7,0.8,0.9,0.95")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("not_bundled/independent_hvg_encoder_marker_benchmark"),  #change path
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(s: str) -> str:
    return str(s).strip().lower()


def parse_list_csv(path: Path, col: str) -> list[str]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}' in {path}")
    vals = df[col].astype(str).str.strip()
    vals = vals[(vals != "") & (vals != "nan")]
    return list(dict.fromkeys(vals.tolist()))


def read_h5ad_csr(path: Path, layer_key: str) -> tuple[sp.csr_matrix, pd.Index, pd.Index]:
    with h5py.File(path, "r") as f:
        if layer_key != "X" and f"layers/{layer_key}" in f:
            g = f[f"layers/{layer_key}"]
        elif "X" in f:
            g = f["X"]
        else:
            raise KeyError(f"Neither layers/{layer_key} nor X found in {path}")
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        obs_raw = f["obs"]["_index"][:]
        var_raw = f["var"]["_index"][:]
    obs_names = pd.Index([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in obs_raw])
    var_names = pd.Index([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in var_raw])
    x = sp.csr_matrix((data, indices, indptr), shape=(len(obs_names), len(var_names)), dtype=np.float32)
    return x, obs_names, var_names


def extract_normlog_features(path: Path, genes: list[str], layer_key: str, target_sum: float) -> tuple[np.ndarray, pd.Index, list[str]]:
    x_counts, obs_names, var_names = read_h5ad_csr(path, layer_key)
    keep = [g for g in genes if g in var_names]
    if not keep:
        raise ValueError(f"No requested genes found in {path}")
    idx = var_names.get_indexer(keep)
    if (idx < 0).any():
        missing = [keep[i] for i in np.where(idx < 0)[0][:10]]
        raise ValueError(f"Missing genes in {path}: {missing}")

    x_sel = x_counts[:, idx].tocsr().astype(np.float32)
    rs = np.asarray(x_counts.sum(axis=1)).ravel().astype(np.float32)
    sf = np.zeros_like(rs, dtype=np.float32)
    nz = rs > 0
    sf[nz] = np.float32(target_sum) / rs[nz]
    x_sel = x_sel.multiply(sf[:, None]).tocsr()
    x_sel.data = np.log1p(x_sel.data, dtype=np.float32)
    x = np.asarray(x_sel.toarray(), dtype=np.float32)
    return x, obs_names, keep


def zscore_ref_fit(x_train: np.ndarray, x_val: np.ndarray, x_query: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.where(std < np.float32(eps), np.float32(1.0), std)
    tr = ((x_train - mean) / std).astype(np.float32)
    va = ((x_val - mean) / std).astype(np.float32)
    q = ((x_query - mean) / std).astype(np.float32)
    return tr, va, q, mean, std, np.where(std == 0, 1.0, std).astype(np.float32)


def infer_hidden_dims_from_state_dict(state_dict: dict[str, torch.Tensor]) -> list[int]:
    pairs: list[tuple[int, int]] = []
    for k, v in state_dict.items():
        m = re.match(r"hidden\.(\d+)\.weight$", k)
        if m is None:
            continue
        if getattr(v, "ndim", None) != 2:
            continue
        pairs.append((int(m.group(1)), int(v.shape[0])))
    pairs.sort(key=lambda x: x[0])
    return [d for _, d in pairs]


def build_encoder_model(ckpt: dict, config: dict) -> encmod.MLP:
    state = ckpt["state_dict"]
    hvg_genes = ckpt.get("hvg_genes", None)
    if hvg_genes is None:
        raise ValueError("Checkpoint missing hvg_genes")
    in_dim = len(hvg_genes)
    out_dim = int(ckpt["latent_dim"])
    hidden = infer_hidden_dims_from_state_dict(state)
    use_ln_in = bool(ckpt.get("input_layernorm", ckpt.get("use_layernorm", False)))
    hidden_ln = bool(ckpt.get("hidden_layernorm", True))
    residual_blocks = int(ckpt.get("residual_blocks", 0))
    dropout = float(config.get("dropout", 0.0))

    model = encmod.MLP(
        in_dim=in_dim,
        hidden=hidden,
        out_dim=out_dim,
        use_layernorm=use_ln_in,
        hidden_layernorm=hidden_ln,
        dropout=dropout,
        residual_blocks=residual_blocks,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def resolve_device(arg: str) -> torch.device:
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda but CUDA is not available")
        return torch.device("cuda")
    if arg == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def project_with_encoder(
    model: encmod.MLP,
    ckpt: dict,
    adata_path: Path,
    batch_size: int,
    device: torch.device,
    layer_key_fallback: str,
) -> tuple[np.ndarray, pd.Index]:
    adata = ad.read_h5ad(adata_path, backed="r")
    obs_names = pd.Index(adata.obs_names.astype(str))
    hvg_genes = [str(g) for g in ckpt["hvg_genes"]]
    var_idx = pd.Index(adata.var_names.astype(str)).get_indexer(hvg_genes)
    if (var_idx < 0).any():
        missing = [hvg_genes[i] for i in np.where(var_idx < 0)[0][:20]]
        raise ValueError(f"Missing HVG genes in {adata_path}: {missing}")

    do_log1p = bool(ckpt.get("input_log1p", True))
    do_norm = bool(ckpt.get("input_normalize", False))
    target_sum = float(ckpt.get("input_target_sum", 1e6))
    use_scale = bool(ckpt.get("input_scale", False))
    scale_mean = np.asarray(ckpt.get("input_scale_mean"), dtype=np.float32) if use_scale else None
    scale_std = np.asarray(ckpt.get("input_scale_std"), dtype=np.float32) if use_scale else None

    layer_key = str(layer_key_fallback)
    if layer_key != "X" and layer_key not in adata.layers:
        raise KeyError(f"Layer '{layer_key}' not found in {adata_path}")

    n = adata.n_obs
    out_dim = int(ckpt["latent_dim"])
    out = np.zeros((n, out_dim), dtype=np.float32)

    model = model.to(device)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = np.arange(start, end, dtype=int)
            x = encmod.fetch_X_batch(
                adata=adata,
                indices=idx,
                hvg_idx=var_idx.astype(int),
                layer_key=layer_key,
                do_log1p=do_log1p,
                do_normalize=do_norm,
                target_sum=target_sum,
                scale_mean=scale_mean,
                scale_std=scale_std,
                mask_hvg=None,
            )
            xb = torch.from_numpy(x).to(device)
            pred = model(xb).cpu().numpy().astype(np.float32, copy=False)
            out[start:end] = pred
    return out, obs_names


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows: list[dict[str, float | str]] = []
    for label, v in rep.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "label": label,
                "precision": float(v.get("precision", 0.0)),
                "recall": float(v.get("recall", 0.0)),
                "f1": float(v.get("f1-score", 0.0)),
                "support": int(v.get("support", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def threshold_curve(y_true: np.ndarray, y_pred: np.ndarray, max_prob: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        assigned = max_prob >= thr
        coverage = float(assigned.mean())
        if assigned.any():
            acc_assigned = float(np.mean(y_pred[assigned] == y_true[assigned]))
        else:
            acc_assigned = float("nan")
        pred_thr = np.where(assigned, y_pred, "unassigned")
        acc_overall_as_wrong = float(np.mean(pred_thr == y_true))
        rows.append(
            {
                "threshold": float(thr),
                "coverage": coverage,
                "assigned_n": int(assigned.sum()),
                "total_n": int(len(assigned)),
                "accuracy_assigned_only": acc_assigned,
                "accuracy_overall_unassigned_as_wrong": acc_overall_as_wrong,
            }
        )
    return pd.DataFrame(rows)


def fit_mlp_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_query: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[base.TrainResult, LabelEncoder, StandardScaler, np.ndarray, np.ndarray]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_query_s = scaler.transform(x_query)

    hidden_dims = [int(x.strip()) for x in args.mlp_hidden_dims.split(",") if x.strip()]
    res = base.train_mlp(
        x_train=x_train_s,
        y_train=y_train_enc,
        x_val=x_val_s,
        y_val=y_val_enc,
        in_dim=x_train_s.shape[1],
        out_dim=len(le.classes_),
        hidden_dims=hidden_dims,
        dropout=float(args.mlp_dropout),
        lr=float(args.mlp_lr),
        batch_size=int(args.mlp_batch_size),
        max_epochs=int(args.mlp_max_epochs),
        label_smoothing=float(args.mlp_label_smoothing),
        patience=int(args.mlp_patience),
        min_delta=float(args.mlp_min_delta),
        weight_decay=float(args.mlp_weight_decay),
        device=device,
    )

    proba_q = base.predict_proba_mlp(
        model=res.model,
        x=x_query_s,
        batch_size=int(args.mlp_batch_size),
        device=device,
    )
    pred_q = le.inverse_transform(np.argmax(proba_q, axis=1))
    max_prob = np.max(proba_q, axis=1)
    return res, le, scaler, pred_q, max_prob


def fit_lgb_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_query: np.ndarray,
    args: argparse.Namespace,
) -> tuple[lgb.LGBMClassifier, LabelEncoder, np.ndarray, np.ndarray]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        learning_rate=float(args.lgb_learning_rate),
        extra_trees=True,
        n_estimators=int(args.lgb_n_estimators),
        num_leaves=int(args.lgb_num_leaves),
        min_child_samples=int(args.lgb_min_child_samples),
        colsample_bytree=float(args.lgb_colsample_bytree),
        subsample=float(args.lgb_subsample),
        random_state=int(args.seed),
        n_jobs=int(args.lgb_n_jobs),
    )
    model.fit(
        x_train,
        y_train_enc,
        eval_set=[(x_val, y_val_enc)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(int(args.lgb_early_stopping_rounds), verbose=False)],
    )
    proba_q = model.predict_proba(x_query)
    pred_q = le.inverse_transform(np.argmax(proba_q, axis=1))
    max_prob = np.max(proba_q, axis=1)
    return model, le, pred_q, max_prob


def save_model_result(
    out_dir: Path,
    model_name: str,
    query_obs: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_prob: np.ndarray,
    thresholds: list[float],
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = basic_metrics(y_true, y_pred)
    per_cls = per_class_metrics(y_true, y_pred)
    curve = threshold_curve(y_true, y_pred, max_prob, thresholds)

    pred_df = pd.DataFrame(
        {
            "cell_id": query_obs,
            "y_true": y_true,
            "pred": y_pred,
            "max_prob": max_prob,
            f"pred_thr_{thresholds[0]:.2f}": np.where(max_prob >= thresholds[0], y_pred, "unassigned"),
        }
    )
    pred_df.to_csv(out_dir / "query_predictions.csv", index=False)
    per_cls.to_csv(out_dir / "per_class_metrics.csv", index=False)
    curve.to_csv(out_dir / "threshold_curve.csv", index=False)
    (out_dir / "metrics_overall.json").write_text(json.dumps(metrics, indent=2))

    thr_target = thresholds[thresholds.index(0.8)] if 0.8 in thresholds else float(0.8)
    row_thr = curve[np.isclose(curve["threshold"].to_numpy(), thr_target)]
    if row_thr.empty:
        row_thr = curve.iloc[[int(np.argmin(np.abs(curve["threshold"].to_numpy() - thr_target)))] ]
    thr_info = row_thr.iloc[0].to_dict()

    return {
        "model": model_name,
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "threshold": float(thr_info["threshold"]),
        "coverage": float(thr_info["coverage"]),
        "assigned_n": int(thr_info["assigned_n"]),
        "total_n": int(thr_info["total_n"]),
        "accuracy_assigned_only": float(thr_info["accuracy_assigned_only"]),
    }


def high_conf_overlap(
    model_preds: dict[str, pd.DataFrame],
    threshold: float,
) -> pd.DataFrame:
    names = sorted(model_preds.keys())
    rows: list[dict[str, float | int | str]] = []
    for a, b in combinations(names, 2):
        da = model_preds[a]
        db = model_preds[b]
        if not da["cell_id"].equals(db["cell_id"]):
            raise ValueError(f"Cell order mismatch between {a} and {b}")

        pa = da["pred"].to_numpy().astype(str)
        pb = db["pred"].to_numpy().astype(str)
        yt = da["y_true"].to_numpy().astype(str)
        ma = da["max_prob"].to_numpy()
        mb = db["max_prob"].to_numpy()

        sa = ma >= threshold
        sb = mb >= threshold
        inter = sa & sb
        union = sa | sb
        only_a = sa & (~sb)
        only_b = sb & (~sa)

        inter_n = int(inter.sum())
        union_n = int(union.sum())
        agree_inter = float(np.mean(pa[inter] == pb[inter])) if inter_n > 0 else float("nan")
        acc_only_a = float(np.mean(pa[only_a] == yt[only_a])) if only_a.any() else float("nan")
        acc_only_b = float(np.mean(pb[only_b] == yt[only_b])) if only_b.any() else float("nan")

        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "threshold": float(threshold),
                "a_highconf_n": int(sa.sum()),
                "b_highconf_n": int(sb.sum()),
                "intersection_n": inter_n,
                "union_n": union_n,
                "jaccard": float(inter_n / union_n) if union_n > 0 else float("nan"),
                "only_a_n": int(only_a.sum()),
                "only_b_n": int(only_b.sum()),
                "agree_rate_on_intersection": agree_inter,
                "acc_only_a": acc_only_a,
                "acc_only_b": acc_only_b,
            }
        )
    return pd.DataFrame(rows)


def load_labels(path: Path, label_col: str) -> tuple[pd.Series, pd.Index]:
    a = ad.read_h5ad(path, backed="r")
    if label_col not in a.obs:
        raise KeyError(f"Missing label column {label_col} in {path}")
    y = a.obs[label_col].astype(str)
    obs = pd.Index(a.obs_names.astype(str))
    return y, obs


def filter_known_classes(y: pd.Series, exclude_norm: set[str], known_classes: set[str] | None = None) -> pd.Series:
    keep = np.array([normalize_label(v) not in exclude_norm for v in y.to_numpy()], dtype=bool)
    if known_classes is not None:
        keep &= y.isin(list(known_classes)).to_numpy()
    return y.loc[keep]


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(x.strip()) for x in str(args.threshold_grid).split(",") if x.strip()]
    if not thresholds:
        thresholds = [float(args.threshold)]
    if float(args.threshold) not in thresholds:
        thresholds.append(float(args.threshold))
    thresholds = sorted(set(thresholds))

    device = resolve_device(str(args.device))
    exclude_norm = {normalize_label(x) for x in str(args.exclude_labels).split(",") if x.strip()}
    methods = [x.strip() for x in str(args.encoder_methods).split(",") if x.strip()]

    # Labels and masks (same protocol across all models).
    y_train_all, obs_train = load_labels(args.train_h5ad, args.label_col)
    y_val_all, obs_val = load_labels(args.val_h5ad, args.label_col)
    y_query_all, obs_query = load_labels(args.query_h5ad, args.label_col)

    y_train = filter_known_classes(y_train_all, exclude_norm, known_classes=None)
    known_classes = set(y_train.astype(str).unique().tolist())
    y_val = filter_known_classes(y_val_all, exclude_norm, known_classes=known_classes)
    y_query = filter_known_classes(y_query_all, exclude_norm, known_classes=known_classes)

    train_mask = obs_train.isin(y_train.index)
    val_mask = obs_val.isin(y_val.index)
    query_mask = obs_query.isin(y_query.index)

    print(
        "[INFO] protocol filter: "
        f"train={int(train_mask.sum())}/{len(train_mask)} "
        f"val={int(val_mask.sum())}/{len(val_mask)} "
        f"query={int(query_mask.sum())}/{len(query_mask)}"
    )

    # Gene lists.
    hvg_genes = parse_list_csv(args.hvg_csv, args.hvg_gene_col)
    marker_genes = parse_list_csv(args.marker_csv, args.marker_gene_col)
    union_genes = list(dict.fromkeys(hvg_genes + marker_genes))
    print(
        f"[INFO] genes: hvg={len(hvg_genes)} marker={len(marker_genes)} union={len(union_genes)}"
    )

    # Build marker and union features once.
    print("[INFO] extracting marker features (normalize_total+log1p)")
    mk_train_raw, obs_train_mk, mk_used = extract_normlog_features(
        args.train_h5ad, marker_genes, args.count_layer, args.target_sum
    )
    mk_val_raw, obs_val_mk, mk_used_val = extract_normlog_features(
        args.val_h5ad, mk_used, args.count_layer, args.target_sum
    )
    mk_query_raw, obs_query_mk, mk_used_q = extract_normlog_features(
        args.query_h5ad, mk_used, args.count_layer, args.target_sum
    )
    if len(mk_used_val) != len(mk_used) or len(mk_used_q) != len(mk_used):
        raise ValueError("Marker genes mismatch across train/val/query after intersection")
    if not obs_train_mk.equals(obs_train) or not obs_val_mk.equals(obs_val) or not obs_query_mk.equals(obs_query):
        raise ValueError("obs order mismatch for marker features")

    mk_train_s, mk_val_s, mk_query_s, mk_mean, mk_std, _ = zscore_ref_fit(
        mk_train_raw, mk_val_raw, mk_query_raw, args.zscore_eps
    )

    print("[INFO] extracting HVG+marker(union) features for Line2 LightGBM")
    union_train_raw, obs_train_u, union_used = extract_normlog_features(
        args.train_h5ad, union_genes, args.count_layer, args.target_sum
    )
    union_val_raw, obs_val_u, union_used_val = extract_normlog_features(
        args.val_h5ad, union_used, args.count_layer, args.target_sum
    )
    union_query_raw, obs_query_u, union_used_q = extract_normlog_features(
        args.query_h5ad, union_used, args.count_layer, args.target_sum
    )
    if len(union_used_val) != len(union_used) or len(union_used_q) != len(union_used):
        raise ValueError("Union genes mismatch across train/val/query after intersection")
    if not obs_train_u.equals(obs_train) or not obs_val_u.equals(obs_val) or not obs_query_u.equals(obs_query):
        raise ValueError("obs order mismatch for union features")

    # Apply protocol masks.
    mk_train = mk_train_s[np.asarray(train_mask, dtype=bool)]
    mk_val = mk_val_s[np.asarray(val_mask, dtype=bool)]
    mk_query = mk_query_s[np.asarray(query_mask, dtype=bool)]
    union_train = union_train_raw[np.asarray(train_mask, dtype=bool)]
    union_val = union_val_raw[np.asarray(val_mask, dtype=bool)]
    union_query = union_query_raw[np.asarray(query_mask, dtype=bool)]

    y_train_np = y_train.astype(str).to_numpy()
    y_val_np = y_val.astype(str).to_numpy()
    y_query_np = y_query.astype(str).to_numpy()
    query_obs_kept = obs_query[np.asarray(query_mask, dtype=bool)]

    model_rows: list[dict[str, float | int | str]] = []
    model_pred_tables: dict[str, pd.DataFrame] = {}

    models_root = args.out_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    # Line1: 4 encoder variants, latent + marker -> MLP.
    for method in methods:
        model_name = f"line1_hvgenc_{method}_plus_marker_mlp"
        print(f"[INFO] running {model_name}")
        method_dir = args.encoder_root / f"hvg_only_{method}"
        model_dir = method_dir / "model"
        if not model_dir.exists():
            raise FileNotFoundError(f"Encoder model dir missing: {model_dir}")

        ckpt_name = "best_encoder.pt" if args.use_best_checkpoint else "encoder.pt"
        ckpt_path = model_dir / ckpt_name
        if not ckpt_path.exists() and args.use_best_checkpoint:
            ckpt_path = model_dir / "encoder.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

        config_path = model_dir / "config.json"
        cfg = {}
        if config_path.exists():
            cfg = json.loads(config_path.read_text())

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        enc = build_encoder_model(ckpt, cfg)

        proj_dir = method_dir / "projection"
        proj_dir.mkdir(parents=True, exist_ok=True)
        train_lat_npy = proj_dir / "ref_train_encoder_latent.npy"
        val_lat_npy = proj_dir / "ref_val_encoder_latent.npy"
        query_lat_npy = proj_dir / "query_encoder_latent.npy"

        if train_lat_npy.exists():
            lat_train = np.load(train_lat_npy)
            obs_train_lat = obs_train
        else:
            lat_train, obs_train_lat = project_with_encoder(
                enc, ckpt, args.train_h5ad, args.projection_batch_size, device, args.count_layer
            )
            np.save(train_lat_npy, lat_train)

        if val_lat_npy.exists():
            lat_val = np.load(val_lat_npy)
            obs_val_lat = obs_val
        else:
            lat_val, obs_val_lat = project_with_encoder(
                enc, ckpt, args.val_h5ad, args.projection_batch_size, device, args.count_layer
            )
            np.save(val_lat_npy, lat_val)

        if query_lat_npy.exists():
            lat_query = np.load(query_lat_npy)
            obs_query_lat = obs_query
        else:
            lat_query, obs_query_lat = project_with_encoder(
                enc, ckpt, args.query_h5ad, args.projection_batch_size, device, args.count_layer
            )
            np.save(query_lat_npy, lat_query)

        if not obs_train_lat.equals(obs_train) or not obs_val_lat.equals(obs_val) or not obs_query_lat.equals(obs_query):
            raise ValueError(f"Latent obs order mismatch for method={method}")

        lat_train = lat_train[np.asarray(train_mask, dtype=bool)]
        lat_val = lat_val[np.asarray(val_mask, dtype=bool)]
        lat_query = lat_query[np.asarray(query_mask, dtype=bool)]

        x_train = np.hstack([lat_train, mk_train]).astype(np.float32, copy=False)
        x_val = np.hstack([lat_val, mk_val]).astype(np.float32, copy=False)
        x_query = np.hstack([lat_query, mk_query]).astype(np.float32, copy=False)

        mlp_res, le, scaler, pred_q, max_prob = fit_mlp_classifier(
            x_train=x_train,
            y_train=y_train_np,
            x_val=x_val,
            y_val=y_val_np,
            x_query=x_query,
            args=args,
            device=device,
        )

        out_dir = models_root / model_name
        row = save_model_result(
            out_dir=out_dir,
            model_name=model_name,
            query_obs=query_obs_kept,
            y_true=y_query_np,
            y_pred=pred_q,
            max_prob=max_prob,
            thresholds=thresholds,
        )
        row["line"] = "line1"
        row["family"] = method
        row["best_epoch"] = int(mlp_res.best_epoch)
        model_rows.append(row)

        torch.save(
            {
                "state_dict": mlp_res.model.state_dict(),
                "hidden_dims": [int(x.strip()) for x in args.mlp_hidden_dims.split(",") if x.strip()],
                "dropout": float(args.mlp_dropout),
                "input_dim": int(x_train.shape[1]),
                "output_dim": int(len(le.classes_)),
                "best_epoch": int(mlp_res.best_epoch),
            },
            out_dir / "mlp_model.pt",
        )
        joblib.dump(le, out_dir / "label_encoder.joblib")
        joblib.dump(scaler, out_dir / "feature_scaler.joblib")

        pred_table = pd.DataFrame(
            {
                "cell_id": query_obs_kept,
                "y_true": y_query_np,
                "pred": pred_q,
                "max_prob": max_prob,
            }
        )
        model_pred_tables[model_name] = pred_table

    # Line2: HVG+marker direct -> LightGBM.
    print("[INFO] running line2_hvg_plus_marker_lgb")
    line2_name = "line2_hvg_plus_marker_lgb"
    lgb_model, le2, pred2, max_prob2 = fit_lgb_classifier(
        x_train=union_train,
        y_train=y_train_np,
        x_val=union_val,
        y_val=y_val_np,
        x_query=union_query,
        args=args,
    )

    out_dir2 = models_root / line2_name
    row2 = save_model_result(
        out_dir=out_dir2,
        model_name=line2_name,
        query_obs=query_obs_kept,
        y_true=y_query_np,
        y_pred=pred2,
        max_prob=max_prob2,
        thresholds=thresholds,
    )
    row2["line"] = "line2"
    row2["family"] = "lgb"
    row2["best_iteration"] = int(getattr(lgb_model, "best_iteration_", -1) or -1)
    model_rows.append(row2)

    joblib.dump(lgb_model, out_dir2 / "lgb_model.joblib")
    joblib.dump(le2, out_dir2 / "label_encoder.joblib")

    pred2_df = pd.DataFrame(
        {
            "cell_id": query_obs_kept,
            "y_true": y_query_np,
            "pred": pred2,
            "max_prob": max_prob2,
        }
    )
    model_pred_tables[line2_name] = pred2_df

    # Global summaries.
    summary = pd.DataFrame(model_rows).sort_values(
        ["macro_f1", "weighted_f1", "accuracy"], ascending=False
    )
    summary.to_csv(args.out_dir / "summary_models.csv", index=False)

    overlap_df = high_conf_overlap(model_pred_tables, float(args.threshold))
    overlap_df.to_csv(args.out_dir / f"highconf_overlap_thr{args.threshold:.2f}.csv", index=False)

    feature_info = {
        "hvg_csv": str(args.hvg_csv),
        "marker_csv": str(args.marker_csv),
        "count_layer": args.count_layer,
        "target_sum": float(args.target_sum),
        "marker_preprocess": "normalize_total(1e6)+log1p+zscore(reftrain-fit)",
        "line2_union_preprocess": "normalize_total(1e6)+log1p",
        "n_hvg_genes": int(len(hvg_genes)),
        "n_marker_genes": int(len(marker_genes)),
        "n_union_genes": int(len(union_genes)),
        "n_marker_used": int(mk_train_raw.shape[1]),
        "n_union_used": int(union_train_raw.shape[1]),
    }
    (args.out_dir / "feature_info.json").write_text(json.dumps(feature_info, indent=2))

    run_cfg = {
        "train_h5ad": str(args.train_h5ad),
        "val_h5ad": str(args.val_h5ad),
        "query_h5ad": str(args.query_h5ad),
        "label_col": args.label_col,
        "exclude_labels": sorted(exclude_norm),
        "encoder_root": str(args.encoder_root),
        "encoder_methods": methods,
        "use_best_checkpoint": bool(args.use_best_checkpoint),
        "thresholds": thresholds,
        "device": str(device),
        "seed": int(args.seed),
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    print("[INFO] done. model summary:")
    print(summary.to_string(index=False))
    print(f"[INFO] saved: {args.out_dir}")


if __name__ == "__main__":
    main()
