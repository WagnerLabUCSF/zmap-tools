#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import benchmark_independent_encoder_marker_hvg as bench
import train_celltype_torch_mlp_lgbxt_ensemble as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train module-gated MLP with encoder latent branch. "
            "Pipeline: HVG(normlog) -> module score (gene x module weight) -> stage soft gate (1x) "
            "-> concat with encoder latent -> train MLP."
        )
    )
    p.add_argument(
        "--train-h5ad",
        type=Path,
        default=Path("not_bundled/ref_train_with_reftrainfit_pca.h5ad"),  #change path
    )
    p.add_argument(
        "--query-h5ad",
        type=Path,
        default=Path("not_bundled/query_with_reftrainfit_pca.h5ad"),  #change path
    )
    p.add_argument(
        "--nicole-h5ad",
        type=Path,
        default=Path("not_bundled/nicole_clean_input.h5ad"),  #change path
    )
    p.add_argument("--train-count-layer", type=str, default="raw_nolog")
    p.add_argument("--query-count-layer", type=str, default="raw_nolog")
    p.add_argument("--nicole-count-layer", type=str, default="raw_counts")
    p.add_argument("--target-sum", type=float, default=1e6)
    p.add_argument("--label-col", type=str, default="ZMAP_CellType")
    p.add_argument("--exclude-labels", type=str, default="blastomere,blastomeres,unassigned")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)

    p.add_argument(
        "--time-model-joblib",
        type=Path,
        default=Path("not_bundled/timeid_hvg_logreg_stage_model.joblib"),  #change path
    )
    p.add_argument(
        "--module-weight-csv",
        type=Path,
        default=Path(
            "not_bundled/matrix/"  #change path
            "A_ref_based_3d_weight_matrix_softSpecificityAllcells_timeblockBg_stageEq_full251209_"
            "excl_blast_unassigned_expr0p25_pct10_tau1_top1bonus1p5_shrinkM50P50_4term/"
            "all_gene_weight_by_celltype_stage_rank_by_weight_mean1_desc_only_MODULE_COMPARABLE_ZEROLOCK.csv"
        ),
    )
    p.add_argument("--stage-col", type=str, default="stage_id")
    p.add_argument("--celltype-col", type=str, default="celltype")
    p.add_argument("--gene-col", type=str, default="gene")
    p.add_argument("--module-weight-col", type=str, default="zlog_weight_mean1_in_module")
    p.add_argument(
        "--stage-order",
        type=str,
        default="Blastula,Gastrula,Segmentation,Pharyngula,Larval",
    )

    p.add_argument(
        "--encoder-dir",
        type=Path,
        default=Path(
            "not_bundled/encoder/"  #change path
            "xpca_harmony_260103_no_unassigned_blastomere_res1_wd1e4_coral0p05"
        ),
    )
    p.add_argument("--projection-batch-size", type=int, default=8192)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")

    p.add_argument(
        "--feature-order",
        type=str,
        choices=["module_latent", "latent_module"],
        default="module_latent",
    )
    p.add_argument(
        "--module-input-scale",
        type=str,
        choices=["none", "zscore"],
        default="none",
        help=(
            "Optional preprocessing for module-branch normlog gene expression before "
            "module score calculation. Time branch is unchanged."
        ),
    )
    p.add_argument("--module-input-scale-eps", type=float, default=1e-6)

    p.add_argument("--hidden-dims", type=str, default="256,128")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--use-layernorm", action="store_true")
    p.add_argument("--tail-res-blocks", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ckpt-use-threshold-lexi", action="store_true")

    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "not_bundled/output/"  #change path
            "time_weighted_module_encoder_mlp_v1"
        ),
    )
    return p.parse_args()


def _parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, thr: float) -> dict[str, float | int]:
    assigned = conf >= float(thr)
    return {
        "threshold": float(thr),
        "coverage": float(np.mean(assigned)),
        "assigned_n": int(assigned.sum()),
        "total_n": int(len(assigned)),
        "accuracy_assigned_only": float(np.mean(y_pred[assigned] == y_true[assigned])) if assigned.any() else float("nan"),
        "accuracy_overall_unassigned_as_wrong": float(np.mean(np.where(assigned, y_pred, "unassigned") == y_true)),
    }


class TimeModel:
    def __init__(self, pkg: dict):
        self.model = pkg["model"]
        self.scaler = pkg["feature_scaler"]
        self.genes = [str(g) for g in pkg["genes_used"]]
        self.time_classes = np.asarray(pkg["label_encoder"].classes_).astype(str)
        self.time_to_stage = {str(k): str(v) for k, v in pkg["time_to_stage"].items()}

    def predict_stage_probs(self, x_gene: np.ndarray, stage_order: list[str]) -> np.ndarray:
        x_s = self.scaler.transform(x_gene)
        p_time = self.model.predict_proba(x_s).astype(np.float32, copy=False)
        s2i = {s: i for i, s in enumerate(stage_order)}
        agg = np.zeros((len(self.time_classes), len(stage_order)), dtype=np.float32)
        for ti, t in enumerate(self.time_classes):
            s = self.time_to_stage.get(str(t), "")
            if s in s2i:
                agg[ti, s2i[s]] = 1.0
        p_stage = p_time @ agg
        rs = np.sum(p_stage, axis=1, keepdims=True)
        rs = np.where(rs > 0, rs, 1.0)
        return p_stage / rs


def read_normlog_dense_for_genes(
    h5ad_path: Path,
    layer_key: str,
    genes: list[str],
    target_sum: float,
) -> tuple[np.ndarray, pd.Index]:
    x_counts, obs_names, var_names = bench.read_h5ad_csr(h5ad_path, layer_key)
    n = x_counts.shape[0]
    g = len(genes)
    out = np.zeros((n, g), dtype=np.float32)

    idx = var_names.get_indexer(genes)
    present = idx >= 0
    if not np.any(present):
        return out, obs_names

    row_sum = np.asarray(x_counts.sum(axis=1)).ravel().astype(np.float32)
    sf = np.zeros_like(row_sum, dtype=np.float32)
    nz = row_sum > 0
    sf[nz] = np.float32(target_sum) / row_sum[nz]

    idx_present = idx[present]
    x_sel = x_counts[:, idx_present].tocsr().astype(np.float32)
    x_sel = x_sel.multiply(sf[:, None]).tocsr()
    if x_sel.nnz:
        x_sel.data = np.log1p(x_sel.data).astype(np.float32)
    out[:, np.where(present)[0]] = np.asarray(x_sel.toarray(), dtype=np.float32)
    return out, obs_names


def fit_zscore_stats(x: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < np.float32(eps), np.float32(1.0), std).astype(np.float32)
    return mean, std


def apply_zscore(x: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if mean is None or std is None:
        return x
    if x.shape[1] != len(mean) or len(mean) != len(std):
        raise ValueError(
            f"module input scale shape mismatch: x={x.shape}, mean={len(mean)}, std={len(std)}"
        )
    return ((x - mean.astype(np.float32, copy=False)) / std.astype(np.float32, copy=False)).astype(
        np.float32,
        copy=False,
    )


def build_module_matrix(
    module_weight_csv: Path,
    stage_col: str,
    celltype_col: str,
    gene_col: str,
    weight_col: str,
    stage_order_hint: list[str],
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(module_weight_csv, usecols=[stage_col, celltype_col, gene_col, weight_col])
    for c in [stage_col, celltype_col, gene_col]:
        df[c] = df[c].astype(str)
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df.dropna(subset=[weight_col]).copy()

    stages_seen = list(dict.fromkeys(df[stage_col].tolist()))
    stage_order = [s for s in stage_order_hint if s in stages_seen]
    stage_order += sorted(set(stages_seen) - set(stage_order))
    if not stage_order:
        raise ValueError("No stage labels found in module weight csv.")

    celltypes = sorted(df[celltype_col].unique().tolist())
    genes = sorted(df[gene_col].unique().tolist())

    modules = [f"{s}@@{c}" for s in stage_order for c in celltypes]
    mod2i = {m: i for i, m in enumerate(modules)}
    g2i = {g: i for i, g in enumerate(genes)}
    s2i = {s: i for i, s in enumerate(stage_order)}

    w_gm = np.zeros((len(genes), len(modules)), dtype=np.float32)
    agg = (
        df.groupby([stage_col, celltype_col, gene_col], as_index=False)[weight_col]
        .mean()
        .reset_index(drop=True)
    )
    for _, r in agg.iterrows():
        s = str(r[stage_col])
        c = str(r[celltype_col])
        g = str(r[gene_col])
        m = f"{s}@@{c}"
        if m in mod2i and g in g2i:
            w_gm[g2i[g], mod2i[m]] = np.float32(r[weight_col])

    module_stage_idx = np.array([s2i[m.split("@@", 1)[0]] for m in modules], dtype=np.int32)
    return genes, modules, stage_order, module_stage_idx, w_gm


def load_encoder(encoder_dir: Path, device_arg: str) -> tuple[torch.nn.Module, dict, torch.device]:
    model_dir = encoder_dir / "model"
    ckpt_path = model_dir / "encoder.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing encoder checkpoint: {ckpt_path}")
    cfg = {}
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    encoder_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder_model = bench.build_encoder_model(encoder_ckpt, cfg)
    device = bench.resolve_device(device_arg)
    return encoder_model, encoder_ckpt, device


def build_features_one_split(
    h5ad_path: Path,
    count_layer: str,
    target_sum: float,
    module_genes: list[str],
    module_w_gm: np.ndarray,
    module_stage_idx: np.ndarray,
    time_model: TimeModel,
    stage_order: list[str],
    encoder_model,
    encoder_ckpt: dict,
    projection_batch_size: int,
    device: torch.device,
    feature_order: str,
    module_input_mean: np.ndarray | None = None,
    module_input_std: np.ndarray | None = None,
) -> tuple[pd.Index, np.ndarray]:
    x_module, obs_mod = read_normlog_dense_for_genes(h5ad_path, count_layer, module_genes, target_sum)
    x_module = apply_zscore(x_module, module_input_mean, module_input_std)
    x_time, obs_time = read_normlog_dense_for_genes(h5ad_path, count_layer, time_model.genes, target_sum)
    if not obs_mod.equals(obs_time):
        raise ValueError(f"{h5ad_path}: obs mismatch between module branch and time branch")

    p_stage = time_model.predict_stage_probs(x_time.astype(np.float32, copy=False), stage_order=stage_order)
    gate = p_stage[:, module_stage_idx]
    module_scores = (x_module @ module_w_gm).astype(np.float32, copy=False)
    module_scores = module_scores * gate.astype(np.float32, copy=False)

    latent, obs_lat = bench.project_with_encoder(
        model=encoder_model,
        ckpt=encoder_ckpt,
        adata_path=h5ad_path,
        batch_size=int(projection_batch_size),
        device=device,
        layer_key_fallback=count_layer,
    )
    if not obs_mod.equals(obs_lat):
        raise ValueError(f"{h5ad_path}: obs mismatch between module branch and encoder branch")

    if feature_order == "module_latent":
        x = np.hstack([module_scores, latent]).astype(np.float32, copy=False)
    else:
        x = np.hstack([latent, module_scores]).astype(np.float32, copy=False)
    return obs_mod, x


def read_label_vec(h5ad_path: Path, label_col: str, obs_order: pd.Index) -> np.ndarray:
    adata = sc.read_h5ad(h5ad_path, backed="r")
    try:
        if label_col not in adata.obs.columns:
            raise KeyError(f"{h5ad_path}: label col not found: {label_col}")
        y = adata.obs[label_col].astype(str).reindex(obs_order).to_numpy()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    return y


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    exclude = {x.strip().lower() for x in str(args.exclude_labels).split(",") if x.strip()}
    stage_order_hint = _parse_csv_list(args.stage_order)

    time_pkg = joblib.load(args.time_model_joblib)
    time_model = TimeModel(time_pkg)
    module_genes, modules, stage_order, module_stage_idx, module_w_gm = build_module_matrix(
        module_weight_csv=args.module_weight_csv,
        stage_col=args.stage_col,
        celltype_col=args.celltype_col,
        gene_col=args.gene_col,
        weight_col=args.module_weight_col,
        stage_order_hint=stage_order_hint,
    )
    encoder_model, encoder_ckpt, device = load_encoder(args.encoder_dir, args.device)

    module_input_mean = None
    module_input_std = None
    module_input_scale_fit_n = 0
    if args.module_input_scale == "zscore":
        x_module_for_scale, obs_scale = read_normlog_dense_for_genes(
            args.train_h5ad,
            args.train_count_layer,
            module_genes,
            float(args.target_sum),
        )
        y_scale = read_label_vec(args.train_h5ad, args.label_col, obs_scale)
        mask_scale = np.array(
            [(str(x).lower() not in exclude) and (str(x).lower() != "nan") for x in y_scale],
            dtype=bool,
        )
        if not mask_scale.any():
            raise RuntimeError("No train cells available to fit module input z-score stats.")
        module_input_scale_fit_n = int(mask_scale.sum())
        module_input_mean, module_input_std = fit_zscore_stats(
            x_module_for_scale[mask_scale],
            eps=float(args.module_input_scale_eps),
        )
        del x_module_for_scale

    # Train split features
    obs_tr, x_tr_all = build_features_one_split(
        h5ad_path=args.train_h5ad,
        count_layer=args.train_count_layer,
        target_sum=float(args.target_sum),
        module_genes=module_genes,
        module_w_gm=module_w_gm,
        module_stage_idx=module_stage_idx,
        time_model=time_model,
        stage_order=stage_order,
        encoder_model=encoder_model,
        encoder_ckpt=encoder_ckpt,
        projection_batch_size=int(args.projection_batch_size),
        device=device,
        feature_order=args.feature_order,
        module_input_mean=module_input_mean,
        module_input_std=module_input_std,
    )
    y_tr_all = read_label_vec(args.train_h5ad, args.label_col, obs_tr)
    mask_tr = np.array([(str(x).lower() not in exclude) and (str(x).lower() != "nan") for x in y_tr_all], dtype=bool)
    x_tr = x_tr_all[mask_tr]
    y_tr = y_tr_all[mask_tr]

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)

    if not (0.0 < float(args.val_frac) < 0.5):
        raise ValueError("--val-frac should be in (0, 0.5).")
    x_train, x_val, y_train, y_val = train_test_split(
        x_tr,
        y_tr_enc,
        test_size=float(args.val_frac),
        random_state=int(args.seed),
        stratify=y_tr_enc,
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)

    hidden_dims = [int(x.strip()) for x in str(args.hidden_dims).split(",") if x.strip()]
    mlp_res = base.train_mlp(
        x_train=x_train_s,
        y_train=y_train,
        x_val=x_val_s,
        y_val=y_val,
        in_dim=x_train_s.shape[1],
        out_dim=len(le.classes_),
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        label_smoothing=float(args.label_smoothing),
        patience=int(args.early_stop_patience),
        min_delta=float(args.early_stop_min_delta),
        weight_decay=float(args.weight_decay),
        device=device,
        use_layernorm=bool(args.use_layernorm),
        tail_res_blocks=int(args.tail_res_blocks),
        ckpt_use_threshold_lexi=bool(args.ckpt_use_threshold_lexi),
        ckpt_threshold=float(args.threshold),
    )

    proba_val = base.predict_proba_mlp(
        model=mlp_res.model,
        x=x_val_s,
        batch_size=int(args.batch_size),
        device=device,
    )
    pred_val = le.inverse_transform(np.argmax(proba_val, axis=1))
    y_val_str = le.inverse_transform(y_val)
    metrics_val = _metrics(y_val_str, pred_val)

    # Query split features
    obs_q, x_q_all = build_features_one_split(
        h5ad_path=args.query_h5ad,
        count_layer=args.query_count_layer,
        target_sum=float(args.target_sum),
        module_genes=module_genes,
        module_w_gm=module_w_gm,
        module_stage_idx=module_stage_idx,
        time_model=time_model,
        stage_order=stage_order,
        encoder_model=encoder_model,
        encoder_ckpt=encoder_ckpt,
        projection_batch_size=int(args.projection_batch_size),
        device=device,
        feature_order=args.feature_order,
        module_input_mean=module_input_mean,
        module_input_std=module_input_std,
    )
    y_q_all = read_label_vec(args.query_h5ad, args.label_col, obs_q)
    x_q_s = scaler.transform(x_q_all)
    proba_q = base.predict_proba_mlp(
        model=mlp_res.model,
        x=x_q_s,
        batch_size=int(args.batch_size),
        device=device,
    )
    pred_q = le.inverse_transform(np.argmax(proba_q, axis=1))
    conf_q = np.max(proba_q, axis=1)

    mask_q = np.array([(str(x).lower() not in exclude) and (str(x).lower() != "nan") for x in y_q_all], dtype=bool)
    metrics_q = _metrics(y_q_all[mask_q], pred_q[mask_q])
    metrics_q = {
        "n_samples_total": int(len(y_q_all)),
        "n_samples_eval": int(mask_q.sum()),
        **metrics_q,
    }
    metrics_q_thr = _threshold_metrics(y_q_all[mask_q], pred_q[mask_q], conf_q[mask_q], float(args.threshold))

    q_pred_df = pd.DataFrame(
        {
            "cell_id": obs_q.astype(str),
            "y_true": y_q_all,
            "pred_label": pred_q,
            "pred_max_proba": conf_q,
            "pred_label_thr": np.where(conf_q >= float(args.threshold), pred_q, "unassigned"),
            "is_assigned_thr": (conf_q >= float(args.threshold)).astype(int),
            "in_eval": mask_q.astype(int),
        }
    )

    # Optional Nicole prediction
    metrics_n = None
    n_pred_df = None
    if args.nicole_h5ad is not None and str(args.nicole_h5ad).strip():
        obs_n, x_n = build_features_one_split(
            h5ad_path=args.nicole_h5ad,
            count_layer=args.nicole_count_layer,
            target_sum=float(args.target_sum),
            module_genes=module_genes,
            module_w_gm=module_w_gm,
            module_stage_idx=module_stage_idx,
            time_model=time_model,
            stage_order=stage_order,
            encoder_model=encoder_model,
            encoder_ckpt=encoder_ckpt,
            projection_batch_size=int(args.projection_batch_size),
            device=device,
            feature_order=args.feature_order,
            module_input_mean=module_input_mean,
            module_input_std=module_input_std,
        )
        x_n_s = scaler.transform(x_n)
        proba_n = base.predict_proba_mlp(
            model=mlp_res.model,
            x=x_n_s,
            batch_size=int(args.batch_size),
            device=device,
        )
        pred_n = le.inverse_transform(np.argmax(proba_n, axis=1))
        conf_n = np.max(proba_n, axis=1)
        assigned_n = conf_n >= float(args.threshold)
        metrics_n = {
            "threshold": float(args.threshold),
            "coverage": float(np.mean(assigned_n)),
            "assigned_n": int(assigned_n.sum()),
            "total_n": int(len(assigned_n)),
            "mean_max_prob": float(np.mean(conf_n)),
            "median_max_prob": float(np.median(conf_n)),
        }
        n_pred_df = pd.DataFrame(
            {
                "cell_id": obs_n.astype(str),
                "pred_label": pred_n,
                "pred_max_proba": conf_n,
                "pred_label_thr": np.where(assigned_n, pred_n, "unassigned"),
                "is_assigned_thr": assigned_n.astype(int),
            }
        )

    # Save outputs
    torch.save(
        {
            "state_dict": mlp_res.model.state_dict(),
            "input_dim": int(x_train_s.shape[1]),
            "output_dim": int(len(le.classes_)),
            "hidden_dims": hidden_dims,
            "dropout": float(args.dropout),
            "use_layernorm": bool(args.use_layernorm),
            "tail_res_blocks": int(args.tail_res_blocks),
            "best_epoch": int(mlp_res.best_epoch),
            "feature_order": args.feature_order,
            "module_weight_col": args.module_weight_col,
            "module_input_scale": args.module_input_scale,
        },
        args.out_dir / "mlp_model.pt",
    )
    joblib.dump(scaler, args.out_dir / "mlp_scaler.joblib")
    joblib.dump(le, args.out_dir / "label_encoder.joblib")
    module_input_scale_mean_path = None
    module_input_scale_std_path = None
    if module_input_mean is not None and module_input_std is not None:
        module_input_scale_mean_path = args.out_dir / "module_input_scale_mean.npy"
        module_input_scale_std_path = args.out_dir / "module_input_scale_std.npy"
        np.save(module_input_scale_mean_path, module_input_mean.astype(np.float32, copy=False))
        np.save(module_input_scale_std_path, module_input_std.astype(np.float32, copy=False))

    q_pred_df.to_csv(args.out_dir / "query_predictions.csv", index=False)
    if n_pred_df is not None:
        n_pred_df.to_csv(args.out_dir / "nicole_predictions.csv", index=False)

    (args.out_dir / "metrics_val.json").write_text(json.dumps(metrics_val, indent=2))
    (args.out_dir / "metrics_query.json").write_text(json.dumps(metrics_q, indent=2))
    (args.out_dir / "metrics_query_threshold.json").write_text(json.dumps(metrics_q_thr, indent=2))
    if metrics_n is not None:
        (args.out_dir / "metrics_nicole.json").write_text(json.dumps(metrics_n, indent=2))

    pd.DataFrame({"module": modules}).to_csv(args.out_dir / "modules_used.csv", index=False)
    pd.DataFrame({"stage_id": stage_order}).to_csv(args.out_dir / "stages_used.csv", index=False)
    pd.DataFrame({"gene": module_genes}).to_csv(args.out_dir / "module_genes_used.csv", index=False)

    run_cfg = {
        "train_h5ad": str(args.train_h5ad),
        "query_h5ad": str(args.query_h5ad),
        "nicole_h5ad": str(args.nicole_h5ad),
        "train_count_layer": args.train_count_layer,
        "query_count_layer": args.query_count_layer,
        "nicole_count_layer": args.nicole_count_layer,
        "target_sum": float(args.target_sum),
        "label_col": args.label_col,
        "exclude_labels": sorted(exclude),
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "time_model_joblib": str(args.time_model_joblib),
        "module_weight_csv": str(args.module_weight_csv),
        "module_weight_col": args.module_weight_col,
        "n_module_genes": int(len(module_genes)),
        "n_modules": int(len(modules)),
        "n_stages": int(len(stage_order)),
        "module_input_scale": args.module_input_scale,
        "module_input_scale_eps": float(args.module_input_scale_eps),
        "module_input_scale_fit_n": int(module_input_scale_fit_n),
        "module_input_scale_mean_npy": str(module_input_scale_mean_path) if module_input_scale_mean_path else None,
        "module_input_scale_std_npy": str(module_input_scale_std_path) if module_input_scale_std_path else None,
        "encoder_dir": str(args.encoder_dir),
        "feature_order": args.feature_order,
        "projection_batch_size": int(args.projection_batch_size),
        "device": str(device),
        "mlp": {
            "hidden_dims": hidden_dims,
            "dropout": float(args.dropout),
            "use_layernorm": bool(args.use_layernorm),
            "tail_res_blocks": int(args.tail_res_blocks),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.max_epochs),
            "label_smoothing": float(args.label_smoothing),
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_min_delta": float(args.early_stop_min_delta),
            "weight_decay": float(args.weight_decay),
            "ckpt_use_threshold_lexi": bool(args.ckpt_use_threshold_lexi),
        },
        "n_train_total": int(len(y_tr_all)),
        "n_train_eval": int(mask_tr.sum()),
        "n_query_total": int(len(y_q_all)),
        "n_query_eval": int(mask_q.sum()),
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    print("[OK] training finished")
    print("[VAL]", json.dumps(metrics_val, ensure_ascii=False))
    print("[QUERY]", json.dumps(metrics_q, ensure_ascii=False))
    print("[QUERY@thr]", json.dumps(metrics_q_thr, ensure_ascii=False))
    if metrics_n is not None:
        print("[NICOLE]", json.dumps(metrics_n, ensure_ascii=False))
    print(f"[OUT] {args.out_dir}")


if __name__ == "__main__":
    main()
