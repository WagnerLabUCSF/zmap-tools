#!/usr/bin/env python3
"""
Train a manual no-tissue ensemble:
  - PyTorch MLP (FastAI-like tabular NN style: dropout + early stopping)
  - LightGBM XT (extra_trees=True)
  - Fixed weighted average of probabilities (default 0.5 / 0.5)

Data:
  - Features: adata.obsm[basis] (default: X_pca_harmony)
  - Label: adata.obs[label_col] (default: ZMAP_CellType)
  - Exclude labels: unassigned, blastomere, blastomeres
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train manual Torch-MLP + LightGBMXT ensemble for celltype classification."
    )
    p.add_argument(
        "--ref-h5ad",
        type=Path,
        default=Path("not_bundled/ZMAP_260103_processed_166k_ref.h5ad"),  #change path
    )
    p.add_argument(
        "--query-h5ad",
        type=Path,
        default=Path("not_bundled/ZMAP_260103_processed_41k_test.h5ad"),  #change path
    )
    p.add_argument(
        "--ref-latent-npy",
        type=Path,
        default=None,
        help="Optional external reference feature matrix (.npy), row-aligned with ref-h5ad obs.",
    )
    p.add_argument(
        "--query-latent-npy",
        type=Path,
        default=None,
        help="Optional external query feature matrix (.npy), row-aligned with query-h5ad obs.",
    )
    p.add_argument("--basis", type=str, default="X_pca_harmony")
    p.add_argument("--label-col", type=str, default="ZMAP_CellType")
    p.add_argument(
        "--ref-keep-ids-csv",
        type=Path,
        default=None,
        help="Optional CSV of reference obs IDs to keep (column 1).",
    )
    p.add_argument(
        "--ref-exclude-ids-csv",
        type=Path,
        default=None,
        help="Optional CSV of reference obs IDs to exclude (column 1).",
    )
    p.add_argument(
        "--exclude-labels",
        type=str,
        default="unassigned,blastomere,blastomeres",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("not_bundled/torch_mlp_lgbxt_notissue"),  #change path
    )
    p.add_argument(
        "--val-size",
        type=int,
        default=2496,
        help="Validation rows from reference set (stratified).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # MLP args (close to prior NeuralNetFastAI settings)
    p.add_argument("--hidden-dims", type=str, default="512,256")
    p.add_argument(
        "--tail-res-blocks",
        type=int,
        default=0,
        help="Residual blocks appended after hidden layers (on last hidden dim).",
    )
    p.add_argument(
        "--use-layernorm",
        action="store_true",
        help="Apply LayerNorm after each hidden Linear layer in MLP.",
    )
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument(
        "--ckpt-use-threshold-lexi",
        action="store_true",
        help=(
            "Select MLP checkpoint by lexicographic rule on validation at --threshold: "
            "constraint accepted_acc>=A, then maximize coverage, then maximize overall@thr."
        ),
    )
    p.add_argument(
        "--ckpt-accepted-acc-min",
        type=float,
        default=0.96,
        help="Fixed A for accepted_acc@threshold constraint. <0 disables fixed floor.",
    )
    p.add_argument(
        "--ckpt-baseline-accepted-acc",
        type=float,
        default=-1.0,
        help="Fallback baseline accepted_acc@threshold. Used when --ckpt-accepted-acc-min<0.",
    )
    p.add_argument(
        "--ckpt-baseline-tol",
        type=float,
        default=0.003,
        help="Tolerance for fallback floor: A = baseline - tol.",
    )

    # LightGBMXT args
    p.add_argument("--lgb-learning-rate", type=float, default=0.05)
    p.add_argument("--lgb-n-estimators", type=int, default=5000)
    p.add_argument("--lgb-num-leaves", type=int, default=255)
    p.add_argument("--lgb-min-child-samples", type=int, default=20)
    p.add_argument("--lgb-colsample-bytree", type=float, default=1.0)
    p.add_argument("--lgb-subsample", type=float, default=1.0)
    p.add_argument("--lgb-early-stopping-rounds", type=int, default=100)
    p.add_argument("--lgb-n-jobs", type=int, default=-1)

    # Ensemble args
    p.add_argument(
        "--mlp-weight",
        type=float,
        default=0.5,
        help="Final probability = mlp_weight * P_mlp + (1-mlp_weight) * P_lgbxt.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="If >=0, report thresholded metrics on query.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("NA").astype(str)


def feature_df(adata, basis: str) -> pd.DataFrame:
    x = adata.obsm[basis]
    cols = [f"pc_{i}" for i in range(x.shape[1])]
    return pd.DataFrame(np.asarray(x), index=adata.obs_names, columns=cols)


def load_feature_matrix(
    adata,
    basis: str,
    latent_npy: Path | None,
    split_name: str,
) -> np.ndarray:
    if latent_npy is None:
        if basis not in adata.obsm:
            raise KeyError(f"{basis} not found in {split_name} .obsm")
        return np.asarray(adata.obsm[basis], dtype=np.float32)

    x = np.load(latent_npy)
    if x.ndim != 2:
        raise ValueError(f"{split_name} latent must be 2D, got shape={x.shape}")
    if x.shape[0] != adata.n_obs:
        raise ValueError(
            f"{split_name} latent row mismatch: latent_rows={x.shape[0]} vs n_obs={adata.n_obs}"
        )
    return np.asarray(x, dtype=np.float32)


def load_labels_obsnames_and_features(
    h5ad_path: Path,
    label_col: str,
    basis: str,
    latent_npy: Path | None,
    split_name: str,
) -> tuple[pd.Series, pd.Index, np.ndarray]:
    if latent_npy is None:
        adata = sc.read_h5ad(h5ad_path)
        if label_col not in adata.obs:
            raise KeyError(f"{label_col} not found in {split_name} .obs")
        y = adata.obs[label_col].astype("string")
        obs_names = pd.Index(adata.obs_names.astype(str))
        x = load_feature_matrix(adata=adata, basis=basis, latent_npy=None, split_name=split_name)
        return y, obs_names, x

    adata = sc.read_h5ad(h5ad_path, backed="r")
    try:
        if label_col not in adata.obs:
            raise KeyError(f"{label_col} not found in {split_name} .obs")
        y = adata.obs[label_col].astype("string").copy()
        obs_names = pd.Index(adata.obs_names.astype(str))
        n_obs = int(adata.n_obs)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    x = np.load(latent_npy)
    if x.ndim != 2:
        raise ValueError(f"{split_name} latent must be 2D, got shape={x.shape}")
    if x.shape[0] != n_obs:
        raise ValueError(f"{split_name} latent row mismatch: latent_rows={x.shape[0]} vs n_obs={n_obs}")
    return y, obs_names, np.asarray(x, dtype=np.float32)


def read_id_set_csv(path: Path) -> set[str]:
    rows: list[list[str]] = []
    with path.open() as f:
        rows = list(csv.reader(f))
    if not rows:
        return set()
    header = rows[0][0].strip().lower() if rows[0] and rows[0][0].strip() else ""
    start = 1 if header in {"cell_id", "id", "obs_name", "obs_names", "index", "_index"} else 0
    out: set[str] = set()
    for row in rows[start:]:
        if row and row[0].strip():
            out.add(row[0].strip())
    return out


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float,
        use_layernorm: bool = False,
        tail_res_blocks: int = 0,
    ):
        super().__init__()
        if tail_res_blocks > 0 and not hidden_dims:
            raise ValueError("tail_res_blocks > 0 requires at least one hidden dim.")
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        for _ in range(tail_res_blocks):
            layers.append(ResidualBlock(prev, dropout))
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return self.act(y + r)


@dataclass
class TrainResult:
    model: MLP
    best_epoch: int
    val_acc: float
    val_loss: float
    val_coverage_thr: float | None = None
    val_acc_assigned_thr: float | None = None
    val_overall_thr: float | None = None
    ckpt_rule: str = "val_acc_loss"


def predict_proba_mlp(
    model: MLP,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            out.append(prob)
    return np.vstack(out)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    in_dim: int,
    out_dim: int,
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    batch_size: int,
    max_epochs: int,
    label_smoothing: float,
    patience: int,
    min_delta: float,
    weight_decay: float,
    device: torch.device,
    use_layernorm: bool = False,
    tail_res_blocks: int = 0,
    ckpt_use_threshold_lexi: bool = False,
    ckpt_threshold: float = 0.8,
    ckpt_accepted_acc_min: float = -1.0,
) -> TrainResult:
    model = MLP(
        in_dim=in_dim,
        hidden_dims=hidden_dims,
        out_dim=out_dim,
        dropout=dropout,
        use_layernorm=use_layernorm,
        tail_res_blocks=tail_res_blocks,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    x_val_t = torch.from_numpy(x_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_cov_thr = float("nan")
    best_acc_assigned_thr = float("nan")
    best_overall_thr = float("nan")
    best_epoch = -1
    best_lexi_score: tuple[float, ...] | None = None
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(x_val_t)
            val_loss = float(criterion(logits_val, y_val_t).item())
            pred_val = torch.argmax(logits_val, dim=1)
            val_acc = float((pred_val == y_val_t).float().mean().item())

        if ckpt_use_threshold_lexi:
            proba_val = torch.softmax(logits_val, dim=1)
            max_prob = torch.max(proba_val, dim=1).values
            assigned = max_prob >= float(ckpt_threshold)
            coverage_thr = float(assigned.float().mean().item())
            if bool(assigned.any().item()):
                acc_assigned_thr = float((pred_val[assigned] == y_val_t[assigned]).float().mean().item())
            else:
                acc_assigned_thr = float("nan")
            overall_thr = float(((pred_val == y_val_t) & assigned).float().mean().item())

            has_floor = ckpt_accepted_acc_min >= 0.0
            meets_floor = (not has_floor) or (
                np.isfinite(acc_assigned_thr) and acc_assigned_thr >= float(ckpt_accepted_acc_min)
            )
            if meets_floor:
                lexi_score = (
                    1.0,
                    coverage_thr,
                    overall_thr,
                    acc_assigned_thr if np.isfinite(acc_assigned_thr) else -1.0,
                    -val_loss,
                )
            else:
                lexi_score = (
                    0.0,
                    acc_assigned_thr if np.isfinite(acc_assigned_thr) else -1.0,
                    coverage_thr,
                    overall_thr,
                    val_acc,
                    -val_loss,
                )
            improved = best_lexi_score is None or lexi_score > best_lexi_score
        else:
            coverage_thr = float("nan")
            acc_assigned_thr = float("nan")
            overall_thr = float("nan")
            improved = (val_acc > best_val_acc + min_delta) or (
                abs(val_acc - best_val_acc) <= min_delta and val_loss < best_val_loss
            )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_cov_thr = coverage_thr
            best_acc_assigned_thr = acc_assigned_thr
            best_overall_thr = overall_thr
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_use_threshold_lexi:
                best_lexi_score = lexi_score
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training failed: no best state captured.")
    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        best_epoch=best_epoch,
        val_acc=best_val_acc,
        val_loss=best_val_loss,
        val_coverage_thr=best_cov_thr if np.isfinite(best_cov_thr) else None,
        val_acc_assigned_thr=best_acc_assigned_thr if np.isfinite(best_acc_assigned_thr) else None,
        val_overall_thr=best_overall_thr if np.isfinite(best_overall_thr) else None,
        ckpt_rule="threshold_lexi" if ckpt_use_threshold_lexi else "val_acc_loss",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 <= args.mlp_weight <= 1.0):
        raise ValueError("--mlp-weight must be in [0, 1]")

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
    exclude = {x.strip() for x in args.exclude_labels.split(",") if x.strip()}

    print(f"[INFO] device={device}")
    print(f"[INFO] loading reference: {args.ref_h5ad}")
    y_ref, ref_obs_names, X_ref_all = load_labels_obsnames_and_features(
        h5ad_path=args.ref_h5ad,
        label_col=args.label_col,
        basis=args.basis,
        latent_npy=args.ref_latent_npy,
        split_name="reference",
    )

    mask_ref = y_ref.notna() & (~y_ref.isin(list(exclude)))
    if args.ref_keep_ids_csv is not None:
        keep_ids = read_id_set_csv(args.ref_keep_ids_csv)
        if not keep_ids:
            raise ValueError(f"--ref-keep-ids-csv is empty: {args.ref_keep_ids_csv}")
        mask_keep = pd.Series(
            ref_obs_names.isin(keep_ids),
            index=ref_obs_names,
        )
        mask_ref = mask_ref & mask_keep
    if args.ref_exclude_ids_csv is not None:
        ex_ids = read_id_set_csv(args.ref_exclude_ids_csv)
        if not ex_ids:
            raise ValueError(f"--ref-exclude-ids-csv is empty: {args.ref_exclude_ids_csv}")
        mask_ex = pd.Series(
            ~ref_obs_names.isin(ex_ids),
            index=ref_obs_names,
        )
        mask_ref = mask_ref & mask_ex
    n_ref_total = len(ref_obs_names)
    n_ref_keep = int(mask_ref.sum())
    print(
        f"[INFO] reference kept={n_ref_keep}/{n_ref_total} "
        f"(removed {n_ref_total - n_ref_keep}; exclude={sorted(exclude)})"
    )

    X_ref = X_ref_all[mask_ref.to_numpy()]
    y_ref_str = as_str(y_ref.loc[mask_ref]).values

    le = LabelEncoder()
    y_ref_enc = le.fit_transform(y_ref_str)

    if args.val_size <= 0 or args.val_size >= len(y_ref_enc):
        raise ValueError(f"Invalid --val-size={args.val_size}; must be in [1, n-1].")

    X_train, X_val, y_train, y_val = train_test_split(
        X_ref,
        y_ref_enc,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_ref_enc,
    )
    print(f"[INFO] split: train={len(y_train)}, val={len(y_val)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    ckpt_acc_floor = float(args.ckpt_accepted_acc_min)
    if ckpt_acc_floor < 0.0 and args.ckpt_baseline_accepted_acc >= 0.0:
        ckpt_acc_floor = float(args.ckpt_baseline_accepted_acc - args.ckpt_baseline_tol)

    print("[INFO] training MLP")
    mlp_res = train_mlp(
        x_train=X_train_scaled,
        y_train=y_train,
        x_val=X_val_scaled,
        y_val=y_val,
        in_dim=X_train.shape[1],
        out_dim=len(le.classes_),
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        use_layernorm=bool(args.use_layernorm),
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        label_smoothing=args.label_smoothing,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        weight_decay=args.weight_decay,
        device=device,
        tail_res_blocks=int(args.tail_res_blocks),
        ckpt_use_threshold_lexi=bool(args.ckpt_use_threshold_lexi),
        ckpt_threshold=float(args.threshold) if args.threshold >= 0 else 0.8,
        ckpt_accepted_acc_min=ckpt_acc_floor,
    )
    print(
        f"[INFO] MLP best_epoch={mlp_res.best_epoch}, "
        f"val_acc={mlp_res.val_acc:.6f}, val_loss={mlp_res.val_loss:.6f}"
    )
    if bool(args.ckpt_use_threshold_lexi):
        print(
            f"[INFO] MLP ckpt(thr={float(args.threshold):.3f}) "
            f"acc_assigned={mlp_res.val_acc_assigned_thr} "
            f"coverage={mlp_res.val_coverage_thr} overall={mlp_res.val_overall_thr} "
            f"floor={ckpt_acc_floor:.6f}"
        )

    print("[INFO] training LightGBMXT")
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        learning_rate=args.lgb_learning_rate,
        extra_trees=True,
        n_estimators=args.lgb_n_estimators,
        num_leaves=args.lgb_num_leaves,
        min_child_samples=args.lgb_min_child_samples,
        colsample_bytree=args.lgb_colsample_bytree,
        subsample=args.lgb_subsample,
        random_state=args.seed,
        n_jobs=args.lgb_n_jobs,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(args.lgb_early_stopping_rounds, verbose=False)],
    )

    proba_mlp_val = predict_proba_mlp(
        model=mlp_res.model,
        x=X_val_scaled,
        batch_size=args.batch_size,
        device=device,
    )
    proba_lgb_val = lgb_model.predict_proba(X_val)
    w = args.mlp_weight
    proba_ens_val = w * proba_mlp_val + (1.0 - w) * proba_lgb_val
    pred_ens_val = le.inverse_transform(np.argmax(proba_ens_val, axis=1))
    y_val_str = le.inverse_transform(y_val)
    val_metrics = metrics(pd.Series(y_val_str), pd.Series(pred_ens_val))
    print(f"[INFO] val metrics (w_mlp={w:.3f}): {val_metrics}")

    print(f"[INFO] loading query: {args.query_h5ad}")
    yq, q_obs_names, X_q_all = load_labels_obsnames_and_features(
        h5ad_path=args.query_h5ad,
        label_col=args.label_col,
        basis=args.basis,
        latent_npy=args.query_latent_npy,
        split_name="query",
    )

    mask_q = yq.notna() & (~yq.isin(list(exclude)))
    n_q_total = len(q_obs_names)
    n_q_keep = int(mask_q.sum())
    print(
        f"[INFO] query kept={n_q_keep}/{n_q_total} "
        f"(removed {n_q_total - n_q_keep}; exclude={sorted(exclude)})"
    )

    q_obs_keep = q_obs_names[mask_q.to_numpy()]
    Xq = X_q_all[mask_q.to_numpy()]
    Xq_scaled = scaler.transform(Xq)
    yq_true = as_str(yq.loc[mask_q])

    proba_mlp_q = predict_proba_mlp(
        model=mlp_res.model,
        x=Xq_scaled,
        batch_size=args.batch_size,
        device=device,
    )
    proba_lgb_q = lgb_model.predict_proba(Xq)
    proba_ens_q = w * proba_mlp_q + (1.0 - w) * proba_lgb_q

    pred_idx = np.argmax(proba_ens_q, axis=1)
    pred_label = le.inverse_transform(pred_idx)
    max_prob = np.max(proba_ens_q, axis=1)

    query_metrics = metrics(yq_true, pd.Series(pred_label, index=q_obs_keep))
    print(f"[INFO] query metrics (no threshold): {query_metrics}")

    pred_out = pd.DataFrame(index=q_obs_keep)
    pred_out.index.name = "cell_id"
    pred_out["y_true"] = yq_true.values
    pred_out["pred_mlp"] = le.inverse_transform(np.argmax(proba_mlp_q, axis=1))
    pred_out["pred_lgbxt"] = le.inverse_transform(np.argmax(proba_lgb_q, axis=1))
    pred_out["pred_ensemble"] = pred_label
    pred_out["pred_max_proba"] = max_prob

    threshold_metrics: dict[str, float] | None = None
    if args.threshold >= 0:
        assigned = max_prob >= args.threshold
        pred_thr = np.where(assigned, pred_label, "unassigned")
        pred_out["pred_ensemble_thr"] = pred_thr

        coverage = float(assigned.mean())
        if assigned.any():
            acc_assigned = float(
                accuracy_score(yq_true[assigned], pd.Series(pred_label, index=q_obs_keep)[assigned])
            )
        else:
            acc_assigned = float("nan")
        acc_overall = float(accuracy_score(yq_true, pd.Series(pred_thr, index=q_obs_keep)))

        threshold_metrics = {
            "threshold": float(args.threshold),
            "coverage": coverage,
            "assigned_n": int(assigned.sum()),
            "total_n": int(len(assigned)),
            "accuracy_assigned_only": acc_assigned,
            "accuracy_overall_unassigned_as_wrong": acc_overall,
        }
        print(f"[INFO] query threshold metrics: {threshold_metrics}")

    torch.save(
        {
            "state_dict": mlp_res.model.state_dict(),
            "input_dim": int(X_train.shape[1]),
            "output_dim": int(len(le.classes_)),
            "hidden_dims": hidden_dims,
            "dropout": float(args.dropout),
            "use_layernorm": bool(args.use_layernorm),
            "tail_res_blocks": int(args.tail_res_blocks),
            "best_epoch": int(mlp_res.best_epoch),
        },
        args.out_dir / "mlp_model.pt",
    )
    joblib.dump(scaler, args.out_dir / "mlp_scaler.joblib")
    joblib.dump(le, args.out_dir / "label_encoder.joblib")
    joblib.dump(lgb_model, args.out_dir / "lgbxt_model.joblib")

    pred_out.to_csv(args.out_dir / "query_predictions.csv")

    with (args.out_dir / "metrics_val.json").open("w") as f:
        json.dump(val_metrics, f, indent=2)
    with (args.out_dir / "metrics_query.json").open("w") as f:
        json.dump(query_metrics, f, indent=2)
    if threshold_metrics is not None:
        with (args.out_dir / "metrics_query_threshold.json").open("w") as f:
            json.dump(threshold_metrics, f, indent=2)

    config = {
        "ref_h5ad": str(args.ref_h5ad),
        "query_h5ad": str(args.query_h5ad),
        "basis": args.basis,
        "ref_latent_npy": str(args.ref_latent_npy) if args.ref_latent_npy else None,
        "query_latent_npy": str(args.query_latent_npy) if args.query_latent_npy else None,
        "ref_keep_ids_csv": str(args.ref_keep_ids_csv) if args.ref_keep_ids_csv else None,
        "ref_exclude_ids_csv": str(args.ref_exclude_ids_csv) if args.ref_exclude_ids_csv else None,
        "label_col": args.label_col,
        "exclude_labels": sorted(exclude),
        "seed": args.seed,
        "device": str(device),
        "mlp": {
            "hidden_dims": hidden_dims,
            "use_layernorm": bool(args.use_layernorm),
            "tail_res_blocks": int(args.tail_res_blocks),
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "label_smoothing": args.label_smoothing,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "weight_decay": args.weight_decay,
            "ckpt_rule": mlp_res.ckpt_rule,
            "ckpt_threshold": float(args.threshold),
            "ckpt_accepted_acc_floor": ckpt_acc_floor,
            "best_epoch_val_coverage_thr": mlp_res.val_coverage_thr,
            "best_epoch_val_acc_assigned_thr": mlp_res.val_acc_assigned_thr,
            "best_epoch_val_overall_thr": mlp_res.val_overall_thr,
        },
        "lgbxt": {
            "learning_rate": args.lgb_learning_rate,
            "n_estimators": args.lgb_n_estimators,
            "num_leaves": args.lgb_num_leaves,
            "min_child_samples": args.lgb_min_child_samples,
            "colsample_bytree": args.lgb_colsample_bytree,
            "subsample": args.lgb_subsample,
            "early_stopping_rounds": args.lgb_early_stopping_rounds,
            "n_jobs": args.lgb_n_jobs,
            "extra_trees": True,
        },
        "ensemble": {
            "mlp_weight": args.mlp_weight,
            "lgbxt_weight": 1.0 - args.mlp_weight,
            "rule": "weighted_average_proba",
        },
    }
    with (args.out_dir / "train_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"[INFO] saved artifacts to: {args.out_dir}")


if __name__ == "__main__":
    main()
