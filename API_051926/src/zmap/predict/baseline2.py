from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


BASELINE2_FILE_DIR = Path(__file__).resolve().parent / "baseline2_file"  #change path
DEFAULT_ENSEMBLE_DIR = BASELINE2_FILE_DIR  #change path
DEFAULT_BASELINE2_MODEL_DIR = (
    DEFAULT_ENSEMBLE_DIR / "baseline2" / "output" / "baseline2_zscore_job1148438"  #change path
)
DEFAULT_BASELINE2_TISSUE_MODEL_DIR = (
    DEFAULT_ENSEMBLE_DIR  #change path
    / "baseline2_tissue"  #change path
    / "output"  #change path
    / "baseline2_tissue_zscore_job1149526"  #change path
)
SHARED_TIME_MODEL_PATH = (  #change path
    BASELINE2_FILE_DIR / "shared" / "time_model" / "timeid_hvg_logreg_stage_model.joblib"  #change path
)
SHARED_ENCODER_DIR = (  #change path
    BASELINE2_FILE_DIR  #change path
    / "shared"  #change path
    / "encoder"  #change path
    / "xpca_harmony_260103_no_unassigned_blastomere_res1_wd1e4_coral0p05"  #change path
)
BASELINE2_MODULE_WEIGHT_CSV = (  #change path
    BASELINE2_FILE_DIR  #change path
    / "baseline2"  #change path
    / "matrix"  #change path
    / "A_ref_based_3d_weight_matrix_softSpecificityAllcells_timeblockBg_stageEq_full251209_excl_blast_unassigned_expr0p25_pct10_tau1_top1bonus1p5_shrinkM50P50_4term"  #change path
    / "all_gene_weight_by_celltype_stage_rank_by_weight_mean1_desc_only_MODULE_COMPARABLE_ZEROLOCK.csv"  #change path
)
BASELINE2_TISSUE_MODULE_WEIGHT_CSV = (  #change path
    BASELINE2_FILE_DIR  #change path
    / "baseline2_tissue"  #change path
    / "matrix"  #change path
    / "A_ref_based_3d_weight_matrix_softSpecificityAllcells_timeblockBg_stageEq_full251209_ZMAP_Tissue_excl_blast_unassigned_expr0p25_pct10_tau1_top1bonus1p5_shrinkM50P50_4term"  #change path
    / "all_gene_weight_by_tissue_stage_rank_by_weight_mean1_desc_only_MODULE_COMPARABLE_ZEROLOCK.csv"  #change path
)

_STAGE_ORDER_DEFAULT = ["Blastula", "Gastrula", "Segmentation", "Pharyngula", "Larval"]
DEFAULT_EVAL_OMIT_LABELS = ("blastomere", "blastomeres", "unassigned", "nan")


def _ensemble_dir() -> Path:
    env_path = os.environ.get("ZMAP_BASELINE2_FILE_DIR", "").strip()  #change path
    if not env_path:  #change path
        env_path = os.environ.get("ZMAP_ENSEMBLE_DIR", "").strip()  #change path
    return Path(env_path).expanduser() if env_path else DEFAULT_ENSEMBLE_DIR  #change path


def _default_model_dir(model: Literal["baseline2", "baseline_tissue"]) -> Path:
    root = _ensemble_dir()  #change path
    if model == "baseline_tissue":
        return (
            root  #change path
            / "baseline2_tissue"  #change path
            / "output"  #change path
            / "baseline2_tissue_zscore_job1149526"  #change path
        )
    return root / "baseline2" / "output" / "baseline2_zscore_job1148438"  #change path


def _script_dir() -> Path:
    env_path = os.environ.get("ZMAP_BASELINE2_SCRIPT_DIR", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return _ensemble_dir() / "baseline2" / "script"  #change path


def _load_ensemble_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """
    Load the original ZMAP/ensemble baseline2 helper modules.

    The trained baseline2 artifacts were created by these scripts, so using
    the same helper code keeps preprocessing, encoder projection, and MLP
    architecture byte-for-byte compatible with the saved checkpoints.
    """
    script_dir = _script_dir()
    if not script_dir.exists():
        raise FileNotFoundError(
            "Could not find baseline2 script directory. "
            "Set ZMAP_BASELINE2_SCRIPT_DIR or ZMAP_ENSEMBLE_DIR. "
            f"Tried: {script_dir}"
        )
    script_dir_s = str(script_dir)
    if script_dir_s not in sys.path:
        sys.path.insert(0, script_dir_s)
    bench = importlib.import_module("benchmark_independent_encoder_marker_hvg")
    base = importlib.import_module("train_celltype_torch_mlp_lgbxt_ensemble")
    twm = importlib.import_module("train_time_weighted_module_encoder_mlp")
    return bench, base, twm


def _available_layers_h5ad(path: Path) -> list[str]:
    import h5py

    with h5py.File(path, "r") as f:
        return list(f["layers"].keys()) if "layers" in f else []


def _infer_count_layer_from_path(path: Path, preferred: str | None) -> str:
    layers = _available_layers_h5ad(path)
    if preferred:
        if preferred == "X" or preferred in layers:
            return preferred
        raise KeyError(
            f"Requested count layer {preferred!r} not found in {path}. "
            f"Available layers: {layers}"
        )
    for candidate in ("raw_nolog", "counts"):
        if candidate in layers:
            return candidate
    return "X"


def _infer_count_layer_from_adata(adata: ad.AnnData, preferred: str | None) -> str:
    if preferred:
        if preferred == "X" or preferred in adata.layers:
            return preferred
        raise KeyError(
            f"Requested count layer {preferred!r} not found in AnnData. "
            f"Available layers: {list(adata.layers.keys())}"
        )
    for candidate in ("raw_nolog", "counts"):
        if candidate in adata.layers:
            return candidate
    return "X"


def _load_run_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run_config.json: {cfg_path}")
    run_cfg = json.loads(cfg_path.read_text())
    bundled_root = BASELINE2_FILE_DIR.resolve()  #change path
    try:
        is_bundled_model = model_dir.resolve().is_relative_to(bundled_root)  #change path
    except AttributeError:
        is_bundled_model = str(model_dir.resolve()).startswith(str(bundled_root))  #change path
    if is_bundled_model:  #change path
        run_cfg["time_model_joblib"] = str(SHARED_TIME_MODEL_PATH)  #change path
        run_cfg["encoder_dir"] = str(SHARED_ENCODER_DIR)  #change path
        run_cfg["module_input_scale_mean_npy"] = str(model_dir / "module_input_scale_mean.npy")  #change path
        run_cfg["module_input_scale_std_npy"] = str(model_dir / "module_input_scale_std.npy")  #change path
        if "baseline2_tissue" in model_dir.parts:  #change path
            run_cfg["module_weight_csv"] = str(BASELINE2_TISSUE_MODULE_WEIGHT_CSV)  #change path
        else:
            run_cfg["module_weight_csv"] = str(BASELINE2_MODULE_WEIGHT_CSV)  #change path
    return run_cfg


def _load_trained_mlp(model_dir: Path, device: torch.device, base: ModuleType):
    import joblib
    import torch

    ckpt_path = model_dir / "mlp_model.pt"
    scaler_path = model_dir / "mlp_scaler.joblib"
    le_path = model_dir / "label_encoder.joblib"
    for path in (ckpt_path, scaler_path, le_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline2 model artifact: {path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = base.MLP(
        in_dim=int(ckpt["input_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        out_dim=int(ckpt["output_dim"]),
        dropout=float(ckpt["dropout"]),
        use_layernorm=bool(ckpt.get("use_layernorm", False)),
        tail_res_blocks=int(ckpt.get("tail_res_blocks", 0)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, joblib.load(scaler_path), joblib.load(le_path), ckpt


def _resolve_hvg_indices(var_names: pd.Index, hvg_genes: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    idx = var_names.get_indexer(hvg_genes)
    lower_map: dict[str, int] = {}
    for i, gene in enumerate(var_names):
        lower_map.setdefault(str(gene).lower(), i)

    match_mode = np.full(len(hvg_genes), "missing", dtype=object)
    resolved_var = np.full(len(hvg_genes), "", dtype=object)
    for i, gene in enumerate(hvg_genes):
        if idx[i] >= 0:
            match_mode[i] = "exact"
            resolved_var[i] = str(var_names[idx[i]])
            continue
        alt = lower_map.get(str(gene).lower(), -1)
        if alt >= 0:
            idx[i] = alt
            match_mode[i] = "casefold"
            resolved_var[i] = str(var_names[alt])

    coverage = pd.DataFrame(
        {
            "hvg_gene": hvg_genes,
            "resolved_var_name": resolved_var,
            "match_mode": match_mode,
            "is_present": idx >= 0,
        }
    )
    return idx, coverage


def _project_with_encoder_allow_missing(
    *,
    model,
    ckpt: dict,
    adata_path: Path,
    batch_size: int,
    device: torch.device,
    layer_key: str,
    bench: ModuleType,
) -> tuple[np.ndarray, pd.Index, pd.DataFrame]:
    import torch

    adata = ad.read_h5ad(adata_path, backed="r")
    try:
        obs_names = pd.Index(adata.obs_names.astype(str))
        var_names = pd.Index(adata.var_names.astype(str))
        hvg_genes = [str(g) for g in ckpt["hvg_genes"]]
        idx, coverage = _resolve_hvg_indices(var_names, hvg_genes)
        present_mask = idx >= 0
        present_idx = idx[present_mask].astype(int)

        do_log1p = bool(ckpt.get("input_log1p", True))
        do_norm = bool(ckpt.get("input_normalize", False))
        target_sum = float(ckpt.get("input_target_sum", 1e6))
        use_scale = bool(ckpt.get("input_scale", False))
        scale_mean = np.asarray(ckpt.get("input_scale_mean"), dtype=np.float32) if use_scale else None
        scale_std = np.asarray(ckpt.get("input_scale_std"), dtype=np.float32) if use_scale else None

        if layer_key != "X" and layer_key not in adata.layers:
            raise KeyError(f"Layer {layer_key!r} not found in {adata_path}")

        layer = bench.encmod.get_layer(adata, layer_key)
        n_obs = int(adata.n_obs)
        latent_dim = int(ckpt["latent_dim"])
        latent = np.zeros((n_obs, latent_dim), dtype=np.float32)

        model = model.to(device)
        with torch.no_grad():
            for start in range(0, n_obs, int(batch_size)):
                end = min(start + int(batch_size), n_obs)
                row_idx = np.arange(start, end, dtype=int)
                x = np.zeros((len(row_idx), len(hvg_genes)), dtype=np.float32)

                if present_idx.size > 0:
                    x_present = layer[row_idx][:, present_idx]
                    if sp.issparse(x_present):
                        x_present = x_present.toarray()
                    x_present = np.asarray(x_present, dtype=np.float32)
                    x[:, present_mask] = x_present

                if do_norm:
                    scale = bench.encmod.compute_scale_factors(
                        adata=adata,
                        indices=row_idx,
                        layer_key=layer_key,
                        target_sum=target_sum,
                    )
                    x = x * scale[:, None]
                if do_log1p:
                    x = np.log1p(x)
                if scale_mean is not None and scale_std is not None:
                    x = (x - scale_mean) / np.where(scale_std == 0, 1.0, scale_std)

                xb = torch.from_numpy(x).to(device)
                latent[start:end] = model(xb).cpu().numpy().astype(np.float32, copy=False)
        return latent, obs_names, coverage
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def _project_with_encoder_allow_missing_adata(
    *,
    model,
    ckpt: dict,
    adata: ad.AnnData,
    batch_size: int,
    device: torch.device,
    layer_key: str,
    bench: ModuleType,
) -> tuple[np.ndarray, pd.Index, pd.DataFrame]:
    import torch

    obs_names = pd.Index(adata.obs_names.astype(str))
    var_names = pd.Index(adata.var_names.astype(str))
    hvg_genes = [str(g) for g in ckpt["hvg_genes"]]
    idx, coverage = _resolve_hvg_indices(var_names, hvg_genes)
    present_mask = idx >= 0
    present_idx = idx[present_mask].astype(int)

    do_log1p = bool(ckpt.get("input_log1p", True))
    do_norm = bool(ckpt.get("input_normalize", False))
    target_sum = float(ckpt.get("input_target_sum", 1e6))
    use_scale = bool(ckpt.get("input_scale", False))
    scale_mean = np.asarray(ckpt.get("input_scale_mean"), dtype=np.float32) if use_scale else None
    scale_std = np.asarray(ckpt.get("input_scale_std"), dtype=np.float32) if use_scale else None

    if layer_key != "X" and layer_key not in adata.layers:
        raise KeyError(f"Layer {layer_key!r} not found in AnnData")

    layer = bench.encmod.get_layer(adata, layer_key)
    n_obs = int(adata.n_obs)
    latent_dim = int(ckpt["latent_dim"])
    latent = np.zeros((n_obs, latent_dim), dtype=np.float32)

    model = model.to(device)
    with torch.no_grad():
        for start in range(0, n_obs, int(batch_size)):
            end = min(start + int(batch_size), n_obs)
            row_idx = np.arange(start, end, dtype=int)
            x = np.zeros((len(row_idx), len(hvg_genes)), dtype=np.float32)

            if present_idx.size > 0:
                x_present = layer[row_idx][:, present_idx]
                if sp.issparse(x_present):
                    x_present = x_present.toarray()
                x_present = np.asarray(x_present, dtype=np.float32)
                x[:, present_mask] = x_present

            if do_norm:
                scale = bench.encmod.compute_scale_factors(
                    adata=adata,
                    indices=row_idx,
                    layer_key=layer_key,
                    target_sum=target_sum,
                )
                x = x * scale[:, None]
            if do_log1p:
                x = np.log1p(x)
            if scale_mean is not None and scale_std is not None:
                x = (x - scale_mean) / np.where(scale_std == 0, 1.0, scale_std)

            xb = torch.from_numpy(x).to(device)
            latent[start:end] = model(xb).cpu().numpy().astype(np.float32, copy=False)

    return latent, obs_names, coverage


def _infer_module_label_col(module_weight_csv: Path, run_cfg: dict) -> str:
    cols = set(pd.read_csv(module_weight_csv, nrows=0).columns.astype(str))
    label_col = str(run_cfg.get("label_col", ""))
    if label_col == "ZMAP_Tissue" and "tissue" in cols:
        return "tissue"
    if "celltype" in cols:
        return "celltype"
    if "tissue" in cols:
        return "tissue"
    raise ValueError(f"Could not infer module label column from {module_weight_csv}")


def _cfg_value(run_cfg: dict, key: str, default):
    value = run_cfg.get(key)
    return default if value is None or value == "" else value


def _read_normlog_dense_for_genes_adata(
    adata: ad.AnnData,
    layer_key: str,
    genes: list[str],
    target_sum: float,
) -> tuple[np.ndarray, pd.Index]:
    if layer_key == "X":
        x_counts = adata.X
    else:
        if layer_key not in adata.layers:
            raise KeyError(f"Layer {layer_key!r} not found in AnnData")
        x_counts = adata.layers[layer_key]

    obs_names = pd.Index(adata.obs_names.astype(str))
    var_names = pd.Index(adata.var_names.astype(str))
    n = int(adata.n_obs)
    g = len(genes)
    out = np.zeros((n, g), dtype=np.float32)

    idx = var_names.get_indexer(genes)
    present = idx >= 0
    if not np.any(present):
        return out, obs_names

    if sp.issparse(x_counts):
        x_counts_csr = x_counts.tocsr()
        row_sum = np.asarray(x_counts_csr.sum(axis=1)).ravel().astype(np.float32)
        sf = np.zeros_like(row_sum, dtype=np.float32)
        nz = row_sum > 0
        sf[nz] = np.float32(target_sum) / row_sum[nz]

        idx_present = idx[present]
        x_sel = x_counts_csr[:, idx_present].tocsr().astype(np.float32)
        x_sel = x_sel.multiply(sf[:, None]).tocsr()
        if x_sel.nnz:
            x_sel.data = np.log1p(x_sel.data).astype(np.float32)
        out[:, np.where(present)[0]] = np.asarray(x_sel.toarray(), dtype=np.float32)
    else:
        x_counts_arr = np.asarray(x_counts)
        row_sum = np.asarray(x_counts_arr.sum(axis=1)).ravel().astype(np.float32)
        sf = np.zeros_like(row_sum, dtype=np.float32)
        nz = row_sum > 0
        sf[nz] = np.float32(target_sum) / row_sum[nz]

        idx_present = idx[present]
        x_sel = np.asarray(x_counts_arr[:, idx_present], dtype=np.float32)
        x_sel *= sf[:, None]
        np.log1p(x_sel, out=x_sel)
        out[:, np.where(present)[0]] = x_sel

    return out, obs_names


def _build_features(
    *,
    h5ad_path: Path,
    count_layer: str,
    run_cfg: dict,
    mlp_ckpt: dict,
    bench: ModuleType,
    twm: ModuleType,
    projection_batch_size: int | None,
    device: torch.device,
):
    import joblib

    time_model = twm.TimeModel(joblib.load(run_cfg["time_model_joblib"]))
    module_weight_csv = Path(run_cfg["module_weight_csv"])
    module_label_col = _infer_module_label_col(module_weight_csv, run_cfg)
    module_genes, _modules, stage_order, module_stage_idx, module_w_gm = twm.build_module_matrix(
        module_weight_csv=module_weight_csv,
        stage_col=str(_cfg_value(run_cfg, "stage_col", "stage_id")),
        celltype_col=module_label_col,
        gene_col=str(_cfg_value(run_cfg, "gene_col", "gene")),
        weight_col=str(
            mlp_ckpt.get(
                "module_weight_col",
                _cfg_value(run_cfg, "module_weight_col", "zlog_weight_mean1_in_module"),
            )
        ),
        stage_order_hint=[str(x) for x in _cfg_value(run_cfg, "stage_order", _STAGE_ORDER_DEFAULT)],
    )

    encoder_model, encoder_ckpt, _ = twm.load_encoder(
        encoder_dir=Path(run_cfg["encoder_dir"]),
        device_arg=str(device),
    )

    module_input_mean = None
    module_input_std = None
    if str(run_cfg.get("module_input_scale", "none")) == "zscore":
        mean_path = Path(run_cfg.get("module_input_scale_mean_npy") or "")
        std_path = Path(run_cfg.get("module_input_scale_std_npy") or "")
        if not mean_path.exists():
            mean_path = Path(run_cfg.get("model_dir", "")) / "module_input_scale_mean.npy"
        if not std_path.exists():
            std_path = Path(run_cfg.get("model_dir", "")) / "module_input_scale_std.npy"
        if not mean_path.exists() or not std_path.exists():
            raise FileNotFoundError(
                f"Missing module z-score stats: {mean_path}, {std_path}"
            )
        module_input_mean = np.load(mean_path).astype(np.float32)
        module_input_std = np.load(std_path).astype(np.float32)

    x_module, obs_module = twm.read_normlog_dense_for_genes(
        h5ad_path,
        count_layer,
        module_genes,
        float(run_cfg["target_sum"]),
    )
    x_module = twm.apply_zscore(x_module, module_input_mean, module_input_std)
    x_time, obs_time = twm.read_normlog_dense_for_genes(
        h5ad_path,
        count_layer,
        time_model.genes,
        float(run_cfg["target_sum"]),
    )
    if not obs_module.equals(obs_time):
        raise ValueError("obs mismatch between module branch and time branch")

    p_stage = time_model.predict_stage_probs(
        x_time.astype(np.float32, copy=False),
        stage_order=stage_order,
    )
    gate = p_stage[:, module_stage_idx]
    module_scores = (x_module @ module_w_gm).astype(np.float32, copy=False)
    module_scores = module_scores * gate.astype(np.float32, copy=False)

    latent, obs_latent, coverage = _project_with_encoder_allow_missing(
        model=encoder_model,
        ckpt=encoder_ckpt,
        adata_path=h5ad_path,
        batch_size=int(projection_batch_size or run_cfg["projection_batch_size"]),
        device=device,
        layer_key=count_layer,
        bench=bench,
    )
    if not obs_module.equals(obs_latent):
        raise ValueError("obs mismatch between module branch and encoder branch")

    if str(run_cfg["feature_order"]) == "module_latent":
        features = np.hstack([module_scores, latent]).astype(np.float32, copy=False)
    else:
        features = np.hstack([latent, module_scores]).astype(np.float32, copy=False)

    pred_stage = np.asarray(stage_order, dtype=object)[np.argmax(p_stage, axis=1)]
    pred_stage_prob = np.max(p_stage, axis=1)
    return obs_module, features, pred_stage, pred_stage_prob, coverage


def _build_features_from_adata(
    *,
    adata_query: ad.AnnData,
    count_layer: str,
    run_cfg: dict,
    mlp_ckpt: dict,
    bench: ModuleType,
    twm: ModuleType,
    projection_batch_size: int | None,
    device: torch.device,
):
    import joblib

    time_model = twm.TimeModel(joblib.load(run_cfg["time_model_joblib"]))
    module_weight_csv = Path(run_cfg["module_weight_csv"])
    module_label_col = _infer_module_label_col(module_weight_csv, run_cfg)
    module_genes, _modules, stage_order, module_stage_idx, module_w_gm = twm.build_module_matrix(
        module_weight_csv=module_weight_csv,
        stage_col=str(_cfg_value(run_cfg, "stage_col", "stage_id")),
        celltype_col=module_label_col,
        gene_col=str(_cfg_value(run_cfg, "gene_col", "gene")),
        weight_col=str(
            mlp_ckpt.get(
                "module_weight_col",
                _cfg_value(run_cfg, "module_weight_col", "zlog_weight_mean1_in_module"),
            )
        ),
        stage_order_hint=[str(x) for x in _cfg_value(run_cfg, "stage_order", _STAGE_ORDER_DEFAULT)],
    )

    encoder_model, encoder_ckpt, _ = twm.load_encoder(
        encoder_dir=Path(run_cfg["encoder_dir"]),
        device_arg=str(device),
    )

    module_input_mean = None
    module_input_std = None
    if str(run_cfg.get("module_input_scale", "none")) == "zscore":
        mean_path = Path(run_cfg.get("module_input_scale_mean_npy") or "")
        std_path = Path(run_cfg.get("module_input_scale_std_npy") or "")
        if not mean_path.exists():
            mean_path = Path(run_cfg.get("model_dir", "")) / "module_input_scale_mean.npy"
        if not std_path.exists():
            std_path = Path(run_cfg.get("model_dir", "")) / "module_input_scale_std.npy"
        if not mean_path.exists() or not std_path.exists():
            raise FileNotFoundError(
                f"Missing module z-score stats: {mean_path}, {std_path}"
            )
        module_input_mean = np.load(mean_path).astype(np.float32)
        module_input_std = np.load(std_path).astype(np.float32)

    x_module, obs_module = _read_normlog_dense_for_genes_adata(
        adata_query,
        count_layer,
        module_genes,
        float(run_cfg["target_sum"]),
    )
    x_module = twm.apply_zscore(x_module, module_input_mean, module_input_std)
    x_time, obs_time = _read_normlog_dense_for_genes_adata(
        adata_query,
        count_layer,
        time_model.genes,
        float(run_cfg["target_sum"]),
    )
    if not obs_module.equals(obs_time):
        raise ValueError("obs mismatch between module branch and time branch")

    p_stage = time_model.predict_stage_probs(
        x_time.astype(np.float32, copy=False),
        stage_order=stage_order,
    )
    del x_time

    gate = p_stage[:, module_stage_idx]
    module_scores = (x_module @ module_w_gm).astype(np.float32, copy=False)
    del x_module
    module_scores *= gate.astype(np.float32, copy=False)

    latent, obs_latent, coverage = _project_with_encoder_allow_missing_adata(
        model=encoder_model,
        ckpt=encoder_ckpt,
        adata=adata_query,
        batch_size=int(projection_batch_size or run_cfg["projection_batch_size"]),
        device=device,
        layer_key=count_layer,
        bench=bench,
    )
    if not obs_module.equals(obs_latent):
        raise ValueError("obs mismatch between module branch and encoder branch")

    if str(run_cfg["feature_order"]) == "module_latent":
        features = np.hstack([module_scores, latent]).astype(np.float32, copy=False)
    else:
        features = np.hstack([latent, module_scores]).astype(np.float32, copy=False)
    del module_scores, latent

    pred_stage = np.asarray(stage_order, dtype=object)[np.argmax(p_stage, axis=1)]
    pred_stage_prob = np.max(p_stage, axis=1)
    return obs_module, features, pred_stage, pred_stage_prob, coverage


def predict_baseline2(
    input_h5ad: str | os.PathLike,
    *,
    model: Literal["baseline2", "baseline_tissue"] = "baseline2",
    model_dir: str | os.PathLike | None = None,
    count_layer: str | None = None,
    device: str = "cpu",
    projection_batch_size: int | None = None,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the trained ZMAP/ensemble baseline2 classifier on an H5AD file.

    Returns ``(predictions, encoder_hvg_coverage)``. Use :func:`baseline2` or
    :func:`baseline_tissue` when you want predictions written into an AnnData
    object's ``.obs`` directly.
    """
    h5ad_path = Path(input_h5ad).expanduser()
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Input H5AD not found: {h5ad_path}")
    model_dir_path = Path(model_dir).expanduser() if model_dir is not None else _default_model_dir(model)
    run_cfg = _load_run_config(model_dir_path)
    run_cfg["model_dir"] = str(model_dir_path)

    bench, base, twm = _load_ensemble_modules()
    run_device = bench.resolve_device(device)
    count_layer_use = _infer_count_layer_from_path(h5ad_path, count_layer)
    mlp, scaler, label_encoder, mlp_ckpt = _load_trained_mlp(model_dir_path, run_device, base)

    obs, features, pred_stage, pred_stage_prob, coverage = _build_features(
        h5ad_path=h5ad_path,
        count_layer=count_layer_use,
        run_cfg=run_cfg,
        mlp_ckpt=mlp_ckpt,
        bench=bench,
        twm=twm,
        projection_batch_size=projection_batch_size,
        device=run_device,
    )

    features_scaled = scaler.transform(features)
    proba = base.predict_proba_mlp(
        model=mlp,
        x=features_scaled,
        batch_size=int(run_cfg["mlp"]["batch_size"]),
        device=run_device,
    )
    pred_idx = np.argmax(proba, axis=1)
    pred_label = label_encoder.inverse_transform(pred_idx)
    pred_max_proba = np.max(proba, axis=1)
    threshold_use = float(run_cfg.get("threshold", 0.8) if threshold is None else threshold)

    pred = pd.DataFrame(
        {
            "cell_id": obs.astype(str),
            "pred_label": pred_label,
            "pred_max_proba": pred_max_proba,
            "pred_label_thr": np.where(pred_max_proba >= threshold_use, pred_label, "unassigned"),
            "pred_stage": pred_stage,
            "pred_stage_prob": pred_stage_prob,
        }
    )
    pred.attrs["zmap_baseline2"] = {
        "model": model,
        "model_dir": str(model_dir_path),
        "count_layer": count_layer_use,
        "threshold": threshold_use,
        "device": str(run_device),
    }
    pred.attrs["probabilities"] = proba.astype(np.float32, copy=False)
    pred.attrs["classes"] = np.asarray(label_encoder.classes_, dtype=object)
    return pred, coverage


def _run_baseline2_for_adata(
    adata_query: ad.AnnData,
    *,
    model: Literal["baseline2", "baseline_tissue"],
    model_dir: str | os.PathLike | None,
    counts_source: str | None,
    device: str,
    projection_batch_size: int | None,
    threshold: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    count_layer_use = _infer_count_layer_from_adata(adata_query, counts_source)

    model_dir_path = Path(model_dir).expanduser() if model_dir is not None else _default_model_dir(model)
    run_cfg = _load_run_config(model_dir_path)
    run_cfg["model_dir"] = str(model_dir_path)

    bench, base, twm = _load_ensemble_modules()
    run_device = bench.resolve_device(device)
    mlp, scaler, label_encoder, mlp_ckpt = _load_trained_mlp(model_dir_path, run_device, base)

    obs, features, pred_stage, pred_stage_prob, coverage = _build_features_from_adata(
        adata_query=adata_query,
        count_layer=count_layer_use,
        run_cfg=run_cfg,
        mlp_ckpt=mlp_ckpt,
        bench=bench,
        twm=twm,
        projection_batch_size=projection_batch_size,
        device=run_device,
    )

    features_scaled = scaler.transform(features)
    proba = base.predict_proba_mlp(
        model=mlp,
        x=features_scaled,
        batch_size=int(run_cfg["mlp"]["batch_size"]),
        device=run_device,
    )
    del features, features_scaled

    pred_idx = np.argmax(proba, axis=1)
    pred_label = label_encoder.inverse_transform(pred_idx)
    pred_max_proba = np.max(proba, axis=1)
    threshold_use = float(run_cfg.get("threshold", 0.8) if threshold is None else threshold)

    pred = pd.DataFrame(
        {
            "cell_id": obs.astype(str),
            "pred_label": pred_label,
            "pred_max_proba": pred_max_proba,
            "pred_label_thr": np.where(pred_max_proba >= threshold_use, pred_label, "unassigned"),
            "pred_stage": pred_stage,
            "pred_stage_prob": pred_stage_prob,
        }
    )
    pred.attrs["zmap_baseline2"] = {
        "model": model,
        "model_dir": str(model_dir_path),
        "count_layer": count_layer_use,
        "threshold": threshold_use,
        "device": str(run_device),
        "input_mode": "AnnData",
    }
    pred.attrs["probabilities"] = proba.astype(np.float32, copy=False)
    pred.attrs["classes"] = np.asarray(label_encoder.classes_, dtype=object)
    return pred, coverage, count_layer_use


def _base_col(label_space: str, label_suffix: str | None) -> str:
    if label_suffix is not None and str(label_suffix) != "":
        return f"{label_space}_{label_suffix}"
    return label_space


def _pctiles(values: pd.Series | np.ndarray) -> dict[str, float] | None:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _string_category_for_h5ad(values) -> pd.Categorical:
    out = pd.Series(values, dtype=object)
    missing = pd.isna(out)
    out.loc[~missing] = out.loc[~missing].map(str)
    out.loc[missing] = np.nan
    return pd.Categorical(out)


def _string_series_for_h5ad(values, *, fillna: str = "unassigned") -> pd.Series:
    out = pd.Series(values, dtype=object)
    missing = pd.isna(out)
    out.loc[~missing] = out.loc[~missing].map(str)
    out.loc[missing] = str(fillna)
    return out.astype(object)


def _write_label_transfer_outputs(
    adata_query: ad.AnnData,
    pred: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    model: Literal["baseline2", "baseline_tissue"],
    ref_label_col: str,
    label_space: str,
    label_suffix: str | None,
    query_truth_col: str | None,
    counts_source: str,
    p_thresh: float | None,
    min_cells_per_label: int | None,
    apply_filters: bool,
    restrict_to_truth_labels: bool,
    time_labels: str | None,
) -> None:
    labels_base = _base_col(label_space, label_suffix)
    col_main = labels_base
    col_unfilt = f"{labels_base}_unfilt"
    col_prob = f"{labels_base}_prob"
    col_dist = f"{labels_base}_dist"
    col_reason = f"{labels_base}_reason"
    col_reject = f"{labels_base}_reject_flag"
    col_rareflag = f"{labels_base}_rare_flag"
    col_probs_mat = f"{labels_base}_probabilities"

    aligned = pred.set_index("cell_id").reindex(adata_query.obs_names.astype(str))
    raw_pred = aligned["pred_label"].astype(object)
    prob = pd.to_numeric(aligned["pred_max_proba"], errors="coerce")

    adata_query.obs[col_unfilt] = raw_pred.to_numpy()
    adata_query.obs[col_main] = raw_pred.to_numpy()
    adata_query.obs[col_prob] = prob.to_numpy(dtype=float)
    adata_query.obs[col_dist] = np.nan

    proba = pred.attrs.get("probabilities")
    classes = pred.attrs.get("classes")
    if proba is not None:
        row_order = pd.Index(pred["cell_id"].astype(str))
        pos = row_order.get_indexer(adata_query.obs_names.astype(str))
        out = np.zeros((adata_query.n_obs, np.asarray(proba).shape[1]), dtype=np.float32)
        valid = pos >= 0
        out[valid] = np.asarray(proba, dtype=np.float32)[pos[valid]]
        adata_query.obsm[col_probs_mat] = out
    if classes is not None:
        adata_query.uns.setdefault("zmap_labels", {}).setdefault(label_space, {})
        adata_query.uns["zmap_labels"][label_space]["Classes"] = list(map(str, classes))

    if apply_filters:
        if p_thresh is None:
            accept = pd.Series(True, index=adata_query.obs_names)
            reason = np.array(["auto"] * adata_query.n_obs, dtype=object)
            reason_categories = ["auto"]
        else:
            accept = pd.Series(prob.to_numpy() >= float(p_thresh), index=adata_query.obs_names)
            reason = np.where(accept.to_numpy(), "proba", "none")
            reason_categories = ["proba", "none"]
        adata_query.obs[col_reason] = pd.Categorical(reason, categories=reason_categories)
        adata_query.obs[col_reject] = ~accept.to_numpy()
        adata_query.obs.loc[~accept.to_numpy(), col_main] = pd.NA

    allowed_truth_labels_total = None
    if restrict_to_truth_labels:
        if not query_truth_col or query_truth_col not in adata_query.obs.columns:
            raise ValueError(
                "restrict_to_truth_labels=True requires query_truth_col to be set "
                "and present in adata_query.obs."
            )
        truth_labels = adata_query.obs[query_truth_col].dropna().astype(str)
        allowed_labels = set(truth_labels.unique().tolist())
        allowed_truth_labels_total = len(allowed_labels)

        current_pred = adata_query.obs[col_main]
        in_truth = current_pred.astype(str).isin(allowed_labels) & ~current_pred.isna()
        out_of_truth = ~in_truth & ~current_pred.isna()

        if col_reason not in adata_query.obs:
            adata_query.obs[col_reason] = pd.Categorical(
                np.array(["truth_label"] * adata_query.n_obs, dtype=object),
                categories=["truth_label", "out_of_truth"],
            )
        else:
            reason_values = adata_query.obs[col_reason].astype(object).to_numpy()
            reason_values[out_of_truth.to_numpy()] = "out_of_truth"
            reason_categories = list(map(str, adata_query.obs[col_reason].cat.categories))
            if "out_of_truth" not in reason_categories:
                reason_categories.append("out_of_truth")
            adata_query.obs[col_reason] = pd.Categorical(reason_values, categories=reason_categories)

        if col_reject not in adata_query.obs:
            adata_query.obs[col_reject] = False
        adata_query.obs.loc[out_of_truth, col_reject] = True
        adata_query.obs.loc[out_of_truth, col_main] = pd.NA
        adata_query.uns.setdefault("zmap_labels", {}).setdefault(label_space, {})
        adata_query.uns["zmap_labels"][label_space]["Allowed Truth Labels"] = sorted(allowed_labels)

    rare_info = None
    if min_cells_per_label is not None and min_cells_per_label > 0:
        print(f"Filtering labels with fewer than {min_cells_per_label} assigned cells...")
        label_counts = adata_query.obs[col_main].value_counts(dropna=True)
        rare_labels = label_counts[label_counts < min_cells_per_label].index
        if len(rare_labels) > 0:
            adata_query.obs[col_rareflag] = adata_query.obs[col_main].isin(rare_labels)
            adata_query.obs.loc[adata_query.obs[col_rareflag], col_main] = pd.NA
            adata_query.uns.setdefault("zmap_labels", {}).setdefault(label_space, {})
            adata_query.uns["zmap_labels"][label_space]["Rare Labels"] = list(rare_labels)
            rare_info = {
                "n_rare_labels_filtered": len(rare_labels),
                "labels": list(map(str, rare_labels[:10])),
            }
            print(
                f"Filtered {len(rare_labels)} rare labels: "
                f"{list(rare_labels[:10])}{'...' if len(rare_labels) > 10 else ''}"
            )

    if time_labels:
        time_base = _base_col(f"ZMAP_{time_labels}", label_suffix)
        adata_query.obs[f"{time_base}_unfilt"] = aligned["pred_stage"].to_numpy()
        adata_query.obs[f"{time_base}_prob"] = aligned["pred_stage_prob"].to_numpy()
        if apply_filters or restrict_to_truth_labels:
            keep = ~adata_query.obs[col_main].isna()
            adata_query.obs[time_base] = pd.NA
            adata_query.obs.loc[keep, time_base] = aligned.loc[keep.to_numpy(), "pred_stage"].to_numpy()
        else:
            adata_query.obs[time_base] = aligned["pred_stage"].to_numpy()

    adata_query.obs[col_unfilt] = _string_category_for_h5ad(adata_query.obs[col_unfilt])
    adata_query.obs[col_main] = _string_category_for_h5ad(adata_query.obs[col_main])
    if time_labels:
        adata_query.obs[f"{time_base}_unfilt"] = _string_category_for_h5ad(
            adata_query.obs[f"{time_base}_unfilt"]
        )
        adata_query.obs[time_base] = _string_category_for_h5ad(adata_query.obs[time_base])

    assigned = ~adata_query.obs[col_main].isna()
    n_total = int(adata_query.n_obs)
    n_assigned = int(assigned.sum())
    pct_assigned = round(100.0 * n_assigned / n_total, 2) if n_total else 0.0
    reject_breakdown = None
    if col_reason in adata_query.obs and col_reject in adata_query.obs:
        reject_breakdown = (
            adata_query.obs.loc[adata_query.obs[col_reject].fillna(True).astype(bool), col_reason]
            .astype(str)
            .value_counts()
            .to_dict()
        )

    encoder_hvg_present = int(coverage["is_present"].sum())
    encoder_hvg_total = int(len(coverage))
    encoder_hvg_pct = (
        round(100.0 * encoder_hvg_present / encoder_hvg_total, 2)
        if encoder_hvg_total
        else 0.0
    )

    metadata = dict(pred.attrs.get("zmap_baseline2", {}))
    run_summary = {
        "Data": {
            "query_n_cells": n_total,
            "ref_n_cells": None,
            "ref_basis": None,
            "query_basis": None,
            "basis_dim": None,
            "ref_label_col": ref_label_col,
            "query_truth_col": query_truth_col,
            "label_space": label_space,
            "omit_labels": ["unknown", "nan", "unassigned"],
            "classes_ref_total": int(len(classes)) if classes is not None else None,
            "classes_predicted_total": int(adata_query.obs[col_main].dropna().astype(str).nunique()),
        },
        "Params": {
            "method": model,
            "model_dir": metadata.get("model_dir"),
            "counts_source": counts_source,
            "p_thresh": p_thresh,
            "d_thresh": None,
            "apply_filters": bool(apply_filters),
            "restrict_to_truth_labels": bool(restrict_to_truth_labels),
            "allowed_truth_labels_total": allowed_truth_labels_total,
            "device_used": metadata.get("device"),
        },
        "Diagnostics": {
            "probability_summary_unfiltered": _pctiles(adata_query.obs[col_prob]),
            "neighbor_distance_summary": None,
            "encoder_hvg_present": encoder_hvg_present,
            "encoder_hvg_total": encoder_hvg_total,
            "encoder_hvg_pct_present": encoder_hvg_pct,
        },
        "Coverage": {
            "n_total": n_total,
            "n_assigned": n_assigned,
            "pct_assigned": pct_assigned,
            "n_rejected": n_total - n_assigned,
            "pct_rejected": round(100.0 - pct_assigned, 2),
            "rejection_breakdown": reject_breakdown,
            "rare_label_filter": rare_info,
        },
    }

    cell_annotations = pd.DataFrame(
        {
            "cell_id": adata_query.obs_names.astype(str),
            col_main: _string_series_for_h5ad(adata_query.obs[col_main]).to_numpy(),
            col_prob: adata_query.obs[col_prob].to_numpy(),
        },
        index=adata_query.obs_names,
    )

    adata_query.uns.setdefault("zmap_labels", {}).setdefault(label_space, {})
    adata_query.uns["zmap_labels"]["_last_space"] = label_space
    adata_query.uns["zmap_labels"][label_space]["Run Summary"] = run_summary
    adata_query.uns["zmap_labels"][label_space]["Cell Annotations"] = cell_annotations
    adata_query.uns["zmap_labels"][label_space]["Encoder HVG Coverage"] = coverage
    adata_query.uns["zmap_labels"][label_space]["_run_config"] = {
        "method": model,
        "ref_label_col": ref_label_col,
        "label_space": label_space,
        "label_suffix": label_suffix,
        "counts_source": counts_source,
        "p_thresh": p_thresh,
        "apply_filters": bool(apply_filters),
        "restrict_to_truth_labels": bool(restrict_to_truth_labels),
        "model_dir": metadata.get("model_dir"),
    }

    print("Predictions complete.")
    print(
        "Encoder HVGs found in query: "
        f"{encoder_hvg_present} / {encoder_hvg_total} ({encoder_hvg_pct:.2f}%)."
    )
    if apply_filters:
        if p_thresh is None:
            print("QC skipped: p_thresh=None -> accepting all cells.")
        else:
            print(f"QC applied with active rules: probability >= {p_thresh}.")
    else:
        print("QC filters disabled; accepting all raw predictions.")
    print(f"{n_assigned} accepted / {n_total} total ({n_total - n_assigned} rejected).")


def _evaluate_label_predictions(
    adata_query: ad.AnnData,
    *,
    label_space: str,
    label_suffix: str | None,
    query_truth_col: str | None,
    apply_filters: bool,
    eval_min_cells_per_label: int | None,
    eval_label_map: dict[str, str] | None,
    eval_truth_label_map: dict[str, str] | None,
    eval_omit_labels: list[str] | tuple[str, ...] | None,
    eval_restrict_to_truth_labels: bool,
    eval_truth_min_cells_per_label: int | None,
    eval_group_col: str | None,
    eval_group_min_cells: int | None,
    plot_eval_curves: bool,
) -> None:
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
        roc_curve,
    )

    labels_base = _base_col(label_space, label_suffix)
    col_main = labels_base
    col_unfilt = f"{labels_base}_unfilt"
    col_reject = f"{labels_base}_reject_flag"
    col_probs_mat = f"{labels_base}_probabilities"
    space = label_space

    if not query_truth_col or query_truth_col not in adata_query.obs.columns:
        print(f"Evaluation skipped: ground-truth column '{query_truth_col}' not found in adata_query.obs.")
        print(f"Finished predicting and annotating: {space}")
        return

    has_custom_eval_filters = any(
        [
            eval_min_cells_per_label is not None,
            bool(eval_truth_label_map),
            bool(eval_restrict_to_truth_labels),
            eval_truth_min_cells_per_label is not None,
            bool(eval_group_col),
            eval_group_min_cells is not None,
        ]
    )
    pred_eval_col = (
        col_unfilt
        if has_custom_eval_filters and col_unfilt in adata_query.obs.columns
        else col_main
    )

    print("Evaluating model performance on ACCEPTED predictions only...")
    has_truth = ~adata_query.obs[query_truth_col].isna()
    not_rejected = (
        ~adata_query.obs[col_reject].fillna(True).astype(bool)
        if apply_filters and col_reject in adata_query.obs
        else True
    )
    has_pred = ~adata_query.obs[pred_eval_col].isna()
    base_mask = has_truth & not_rejected & has_pred

    eval_df = pd.DataFrame(
        {
            "truth": adata_query.obs.loc[base_mask, query_truth_col].astype(str).values,
            "pred": adata_query.obs.loc[base_mask, pred_eval_col].astype(str).values,
        },
        index=adata_query.obs_names[base_mask],
    )
    if eval_group_col:
        if eval_group_col not in adata_query.obs.columns:
            raise ValueError(
                f"eval_group_col={eval_group_col!r} not found in adata_query.obs."
            )
        eval_df["group"] = adata_query.obs.loc[base_mask, eval_group_col].astype(str).values

    if eval_label_map:
        eval_df["truth"] = eval_df["truth"].map(
            lambda x: eval_label_map.get(str(x), str(x))
        )
        eval_df["pred"] = eval_df["pred"].map(
            lambda x: eval_label_map.get(str(x), str(x))
        )

    if eval_truth_label_map:
        eval_df["truth"] = eval_df["truth"].map(
            lambda x: eval_truth_label_map.get(str(x), str(x))
        )

    eval_filter_stats: dict[str, object] = {
        "n_base_eval_rows": int(len(eval_df)),
        "prediction_source_col": pred_eval_col,
    }

    if eval_omit_labels:
        omit_set = {str(x).lower() for x in eval_omit_labels}
        truth_lower = eval_df["truth"].astype(str).str.lower()
        omit_mask = truth_lower.isin(omit_set)
        eval_filter_stats["truth_labels_omitted"] = (
            eval_df.loc[omit_mask, "truth"].astype(str).value_counts().to_dict()
        )
        eval_filter_stats["n_truth_labels_omitted"] = int(omit_mask.sum())
        eval_filter_stats["eval_omit_labels"] = list(eval_omit_labels)
        eval_df = eval_df.loc[~omit_mask].copy()

    if eval_restrict_to_truth_labels:
        allowed_truth_labels_initial = set(eval_df["truth"].astype(str).unique().tolist())
        keep_mask = eval_df["pred"].astype(str).isin(allowed_truth_labels_initial)
        eval_filter_stats["n_removed_pred_not_in_truth_initial"] = int((~keep_mask).sum())
        eval_filter_stats["removed_pred_not_in_truth_initial"] = (
            eval_df.loc[~keep_mask, "pred"].astype(str).value_counts().to_dict()
        )
        eval_df = eval_df.loc[keep_mask].copy()

    dropped_truth_labels: list[str] = []
    if eval_truth_min_cells_per_label is not None and eval_truth_min_cells_per_label > 0:
        truth_counts = eval_df["truth"].astype(str).value_counts()
        min_n = int(eval_truth_min_cells_per_label)
        keep_truth = truth_counts[truth_counts >= min_n].index.astype(str)
        dropped_truth = truth_counts[truth_counts < min_n]
        dropped_truth_labels = list(map(str, dropped_truth.index.tolist()))
        eval_filter_stats["truth_labels_filtered_by_min_cells"] = dropped_truth.to_dict()
        eval_df = eval_df.loc[eval_df["truth"].astype(str).isin(set(keep_truth))].copy()

        if eval_restrict_to_truth_labels:
            keep_truth_set = set(keep_truth.tolist())
            refilter_mask = eval_df["pred"].astype(str).isin(keep_truth_set)
            eval_filter_stats["n_removed_pred_not_in_truth_refilter"] = int((~refilter_mask).sum())
            eval_filter_stats["removed_pred_not_in_truth_refilter"] = (
                eval_df.loc[~refilter_mask, "pred"].astype(str).value_counts().to_dict()
            )
            eval_df = eval_df.loc[refilter_mask].copy()

    eval_excluded_rare_labels: list[str] = []
    eval_excluded_rare_n = 0
    if eval_min_cells_per_label is not None and eval_min_cells_per_label > 0:
        eval_counts = eval_df["pred"].astype(str).value_counts()
        rare_eval_labels = eval_counts[
            eval_counts < int(eval_min_cells_per_label)
        ].index.astype(str)
        if len(rare_eval_labels) > 0:
            rare_eval_mask = eval_df["pred"].astype(str).isin(rare_eval_labels)
            eval_excluded_rare_n = int(rare_eval_mask.sum())
            eval_excluded_rare_labels = list(map(str, rare_eval_labels.tolist()))
            eval_df = eval_df.loc[~rare_eval_mask].copy()

    eval_excluded_rare_groups: list[str] = []
    eval_excluded_rare_group_n = 0
    if (
        eval_group_col
        and eval_group_min_cells is not None
        and eval_group_min_cells > 0
        and "group" in eval_df.columns
    ):
        group_counts = eval_df["group"].astype(str).value_counts()
        rare_groups = group_counts[group_counts < int(eval_group_min_cells)].index.astype(str)
        if len(rare_groups) > 0:
            rare_group_mask = eval_df["group"].astype(str).isin(rare_groups)
            eval_excluded_rare_group_n = int(rare_group_mask.sum())
            eval_excluded_rare_groups = list(map(str, rare_groups.tolist()))
            eval_filter_stats["groups_filtered_by_min_cells"] = (
                eval_df.loc[rare_group_mask, "group"].astype(str).value_counts().to_dict()
            )
            eval_df = eval_df.loc[~rare_group_mask].copy()

    n_eval = int(len(eval_df))
    if n_eval == 0:
        print("No accepted rows available for evaluation after filtering; metrics not computed.")
        print(f"Finished predicting and annotating: {space}")
        return

    eval_index = pd.Index(eval_df.index)
    eval_mask = adata_query.obs_names.isin(eval_index)
    true_labels_values = eval_df["truth"].astype(str).values
    predicted_labels_values = eval_df["pred"].astype(str).values
    probabilities_eval_all = adata_query.obsm[col_probs_mat][eval_mask, :]

    model_classes = adata_query.uns.get("zmap_labels", {}).get(space, {}).get("Classes", [])
    class_indices = {str(cls): i for i, cls in enumerate(model_classes)}

    true_classes = set(np.unique(true_labels_values))
    predicted_classes = set(np.unique(predicted_labels_values))
    overlapping_classes = sorted(
        cls for cls in true_classes.intersection(predicted_classes) if cls in class_indices
    )
    if len(overlapping_classes) == 0:
        print("No overlapping classes between true and predicted after filtering; metrics not computed.")
        print(f"Finished predicting and annotating: {space}")
        return

    col_idx = [class_indices[cls] for cls in overlapping_classes]
    probabilities_eval = probabilities_eval_all[:, col_idx]
    y_true_binarized = np.column_stack(
        [(true_labels_values == cls).astype(int) for cls in overlapping_classes]
    )

    per_class = precision_recall_fscore_support(
        true_labels_values,
        predicted_labels_values,
        labels=overlapping_classes,
        zero_division=0,
    )
    cm = confusion_matrix(true_labels_values, predicted_labels_values, labels=overlapping_classes)
    cm_df = pd.DataFrame(cm, index=overlapping_classes, columns=overlapping_classes)

    accuracy = accuracy_score(true_labels_values, predicted_labels_values)
    macro_precision = precision_score(
        true_labels_values, predicted_labels_values, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        true_labels_values, predicted_labels_values, average="macro", zero_division=0
    )
    macro_f1 = f1_score(true_labels_values, predicted_labels_values, average="macro", zero_division=0)

    class_auroc = {}
    for i, label in enumerate(overlapping_classes):
        y_i = y_true_binarized[:, i]
        if np.unique(y_i).size < 2:
            class_auroc[label] = np.nan
            continue
        fpr, tpr, _ = roc_curve(y_i, probabilities_eval[:, i])
        class_auroc[label] = auc(fpr, tpr)
    try:
        macro_auroc = roc_auc_score(
            y_true_binarized, probabilities_eval, average="macro", multi_class="ovr"
        )
    except ValueError:
        macro_auroc = float(np.nanmean(list(class_auroc.values())))

    metrics_dict = {
        "Aggregate Metrics": pd.DataFrame(
            {
                "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Macro AUROC"],
                "Score": [accuracy, macro_precision, macro_recall, macro_f1, macro_auroc],
            }
        ),
        "Class-Specific Metrics": pd.DataFrame(
            {
                "Class": overlapping_classes,
                "Precision": per_class[0],
                "Recall": per_class[1],
                "F1-Score": per_class[2],
                "AUROC": [class_auroc[label] for label in overlapping_classes],
                "Support": per_class[3],
            }
        ),
        "Confusion Matrix": cm_df,
        "Eval N": n_eval,
        "Eval Excluded Rare Prediction N": eval_excluded_rare_n,
        "Eval Excluded Rare Prediction Labels": eval_excluded_rare_labels,
        "Eval Excluded Rare Group N": eval_excluded_rare_group_n,
        "Eval Excluded Rare Groups": eval_excluded_rare_groups,
        "Eval Filter Stats": eval_filter_stats,
    }

    metrics_dict["Run Summary"] = adata_query.uns["zmap_labels"].get(space, {}).get("Run Summary")
    if metrics_dict["Run Summary"] is not None:
        metrics_dict["Run Summary"].setdefault("Params", {})
        metrics_dict["Run Summary"].setdefault("Data", {})
        metrics_dict["Run Summary"].setdefault("Coverage", {})
        metrics_dict["Run Summary"]["Params"]["eval_min_cells_per_label"] = eval_min_cells_per_label
        metrics_dict["Run Summary"]["Params"]["eval_label_map"] = (
            dict(eval_label_map) if eval_label_map else None
        )
        metrics_dict["Run Summary"]["Params"]["eval_restrict_to_truth_labels"] = (
            eval_restrict_to_truth_labels
        )
        metrics_dict["Run Summary"]["Params"]["eval_truth_min_cells_per_label"] = (
            eval_truth_min_cells_per_label
        )
        metrics_dict["Run Summary"]["Params"]["eval_group_col"] = eval_group_col
        metrics_dict["Run Summary"]["Params"]["eval_group_min_cells"] = eval_group_min_cells
        metrics_dict["Run Summary"]["Params"]["eval_truth_label_map"] = (
            dict(eval_truth_label_map) if eval_truth_label_map else None
        )
        metrics_dict["Run Summary"]["Params"]["eval_omit_labels"] = (
            list(eval_omit_labels) if eval_omit_labels else None
        )
        metrics_dict["Run Summary"].setdefault("Coverage", {})
        metrics_dict["Run Summary"]["Coverage"][
            "eval_excluded_rare_prediction_n"
        ] = eval_excluded_rare_n
        metrics_dict["Run Summary"]["Coverage"][
            "eval_excluded_rare_group_n"
        ] = eval_excluded_rare_group_n
        n_total = int(adata_query.n_obs)
        pct_eval = round(100.0 * n_eval / n_total, 2) if n_total else 0.0
        metrics_dict["Run Summary"]["Data"]["classes_predicted_total"] = int(
            eval_df["pred"].astype(str).nunique()
        )
        metrics_dict["Run Summary"]["Data"]["classes_truth_total"] = int(
            eval_df["truth"].astype(str).nunique()
        )
        metrics_dict["Run Summary"]["Coverage"]["n_assigned"] = n_eval
        metrics_dict["Run Summary"]["Coverage"]["pct_assigned"] = pct_eval
        metrics_dict["Run Summary"]["Coverage"]["n_rejected"] = n_total - n_eval
        metrics_dict["Run Summary"]["Coverage"]["pct_rejected"] = round(100.0 - pct_eval, 2)
        metrics_dict["Run Summary"]["Coverage"]["n_total"] = n_total
    adata_query.uns.setdefault("zmap_labels", {})
    adata_query.uns["zmap_labels"][space] = metrics_dict

    print(metrics_dict["Aggregate Metrics"])

    if plot_eval_curves:
        import matplotlib.pyplot as plt

        print("Plotting ROC and PR curves...")
        for i, label in enumerate(overlapping_classes):
            y_i = y_true_binarized[:, i]
            if np.unique(y_i).size < 2:
                continue
            plt.figure(figsize=(8, 4))
            fpr, tpr, _ = roc_curve(y_i, probabilities_eval[:, i])
            precision, recall, _ = precision_recall_curve(y_i, probabilities_eval[:, i])

            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.2f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.title(f"ROC - {label}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(recall, precision)
            plt.title(f"Precision-Recall - {label}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.tight_layout()
            plt.show()


def predict_labels_mlp(
    adata_query: ad.AnnData,
    adata_ref: ad.AnnData | None = None,
    *,
    prediction_level: Literal["celltype", "tissue"] = "celltype",
    ref_label_col: str | None = None,
    label_space: str | None = None,
    query_truth_col: str | None = None,
    counts_source: str | None = "counts",
    query_raw_counts_source: str | None = None,
    label_suffix: str | None = "predicted",
    time_labels: str | None = "stage_id",
    model_dir: str | os.PathLike | None = None,
    device: str = "cpu",
    projection_batch_size: int | None = None,
    p_thresh: float | None = 0.8,
    min_cells_per_label: int | None = 15,
    apply_filters: bool = True,
    restrict_to_truth_labels: bool = False,
    evaluate: bool = False,
    eval_min_cells_per_label: int | None = None,
    eval_label_map: dict[str, str] | None = None,
    eval_truth_label_map: dict[str, str] | None = None,
    eval_omit_labels: list[str] | tuple[str, ...] | None = DEFAULT_EVAL_OMIT_LABELS,
    eval_restrict_to_truth_labels: bool = False,
    eval_truth_min_cells_per_label: int | None = None,
    eval_group_col: str | None = None,
    eval_group_min_cells: int | None = None,
    plot_eval_curves: bool = False,
    return_predictions: bool = False,
):
    """
    MLP label transfer with the same AnnData contract as ``predict_labels_kNN``.

    Set ``prediction_level="celltype"`` to run the CellType MLP, or
    ``prediction_level="tissue"`` to run the tissue MLP. The reference argument
    is accepted for pipeline parity but is not used by the trained MLP models.
    Set ``restrict_to_truth_labels=True`` to mask final predictions that do not
    appear in ``query_truth_col``; raw predictions remain in ``*_unfilt``.
    Use ``eval_label_map`` to merge labels symmetrically on truth and
    prediction during evaluation. Use ``eval_truth_label_map``,
    ``eval_restrict_to_truth_labels``,
    ``eval_truth_min_cells_per_label``, and ``eval_group_col`` /
    ``eval_group_min_cells`` to reproduce custom external benchmark filtering
    without changing the stored prediction outputs.
    Results are written into ``adata_query.obs`` under ``label_space`` and into
    ``adata_query.uns['zmap_labels'][label_space]``.
    """
    del adata_ref
    if prediction_level not in {"celltype", "tissue"}:
        raise ValueError(
            "prediction_level must be either 'celltype' or 'tissue'. "
            f"Got {prediction_level!r}."
        )
    is_tissue = prediction_level == "tissue"
    model = "baseline_tissue" if is_tissue else "baseline2"
    ref_label_col = ref_label_col or ("ZMAP_Tissue" if is_tissue else "ZMAP_CellType")
    space = label_space or ref_label_col
    counts_source = query_raw_counts_source or counts_source
    pred, coverage, count_layer_use = _run_baseline2_for_adata(
        adata_query,
        model=model,
        model_dir=model_dir,
        counts_source=counts_source,
        device=device,
        projection_batch_size=projection_batch_size,
        threshold=p_thresh,
    )
    _write_label_transfer_outputs(
        adata_query,
        pred,
        coverage,
        model=model,
        ref_label_col=ref_label_col,
        label_space=space,
        label_suffix=label_suffix,
        query_truth_col=query_truth_col,
        counts_source=count_layer_use,
        p_thresh=p_thresh,
        min_cells_per_label=min_cells_per_label,
        apply_filters=apply_filters,
        restrict_to_truth_labels=restrict_to_truth_labels,
        time_labels=time_labels,
    )
    if evaluate:
        _evaluate_label_predictions(
            adata_query,
            label_space=space,
            label_suffix=label_suffix,
            query_truth_col=query_truth_col,
            apply_filters=apply_filters,
            eval_min_cells_per_label=eval_min_cells_per_label,
            eval_label_map=eval_label_map,
            eval_truth_label_map=eval_truth_label_map,
            eval_omit_labels=eval_omit_labels,
            eval_restrict_to_truth_labels=eval_restrict_to_truth_labels,
            eval_truth_min_cells_per_label=eval_truth_min_cells_per_label,
            eval_group_col=eval_group_col,
            eval_group_min_cells=eval_group_min_cells,
            plot_eval_curves=plot_eval_curves,
        )
    print(f"Finished predicting and annotating: {space}")
    if return_predictions:
        return pred, coverage
    return None


def predict_labels_baseline2(
    adata_query: ad.AnnData,
    adata_ref: ad.AnnData | None = None,
    **kwargs,
):
    """Backward-compatible wrapper for ``predict_labels_mlp(prediction_level='celltype')``."""
    kwargs.pop("prediction_level", None)
    return predict_labels_mlp(adata_query, adata_ref, prediction_level="celltype", **kwargs)


def predict_labels_baseline_tissue(
    adata_query: ad.AnnData,
    adata_ref: ad.AnnData | None = None,
    **kwargs,
):
    """Backward-compatible wrapper for ``predict_labels_mlp(prediction_level='tissue')``."""
    kwargs.pop("prediction_level", None)
    return predict_labels_mlp(adata_query, adata_ref, prediction_level="tissue", **kwargs)


def _run_on_adata_or_path(
    adata_or_path: ad.AnnData | str | os.PathLike,
    *,
    model: Literal["baseline2", "baseline_tissue"],
    prefix: str,
    model_dir: str | os.PathLike | None,
    count_layer: str | None,
    device: str,
    projection_batch_size: int | None,
    threshold: float | None,
    copy: bool,
    return_predictions: bool,
):
    if isinstance(adata_or_path, ad.AnnData):
        adata = adata_or_path.copy() if copy else adata_or_path
        count_layer_use = _infer_count_layer_from_adata(adata, count_layer)
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            adata.write_h5ad(tmp_path)
            pred, coverage = predict_baseline2(
                tmp_path,
                model=model,
                model_dir=model_dir,
                count_layer=count_layer_use,
                device=device,
                projection_batch_size=projection_batch_size,
                threshold=threshold,
            )
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        metadata = dict(pred.attrs.get("zmap_baseline2", {}))
        metadata["n_encoder_hvg_present"] = int(coverage["is_present"].sum())
        metadata["n_encoder_hvg_model"] = int(len(coverage))
        aligned = pred.set_index("cell_id").reindex(adata.obs_names.astype(str))
        adata.obs[prefix] = aligned["pred_label"].to_numpy()
        adata.obs[f"{prefix}_prob"] = aligned["pred_max_proba"].to_numpy()
        adata.obs[f"{prefix}_thr"] = aligned["pred_label_thr"].to_numpy()
        adata.obs[f"{prefix}_stage"] = aligned["pred_stage"].to_numpy()
        adata.obs[f"{prefix}_stage_prob"] = aligned["pred_stage_prob"].to_numpy()
        adata.uns.setdefault("zmap_baseline2", {})
        adata.uns["zmap_baseline2"][model] = metadata
        if return_predictions:
            return adata, pred, coverage
        return adata

    pred, coverage = predict_baseline2(
        adata_or_path,
        model=model,
        model_dir=model_dir,
        count_layer=count_layer,
        device=device,
        projection_batch_size=projection_batch_size,
        threshold=threshold,
    )
    if return_predictions:
        return pred, coverage
    return pred


def baseline2(
    adata_or_path: ad.AnnData | str | os.PathLike,
    *,
    model_dir: str | os.PathLike | None = None,
    count_layer: str | None = None,
    device: str = "cpu",
    projection_batch_size: int | None = None,
    threshold: float | None = None,
    prefix: str = "ZMAP_CellType_prediction",
    copy: bool = False,
    return_predictions: bool = False,
):
    """
    Run the trained ZMAP/ensemble ``baseline2`` CellType classifier.

    For AnnData input this follows the same output contract as
    ``predict_labels_kNN`` and writes ``ZMAP_CellType_predicted`` plus metadata
    under ``adata.uns['zmap_labels']['ZMAP_CellType']``. For H5AD path input it
    returns the lower-level predictions DataFrame.
    """
    if isinstance(adata_or_path, ad.AnnData):
        adata = adata_or_path.copy() if copy else adata_or_path
        out = predict_labels_mlp(
            adata,
            prediction_level="celltype",
            counts_source=count_layer,
            model_dir=model_dir,
            device=device,
            projection_batch_size=projection_batch_size,
            p_thresh=threshold if threshold is not None else 0.8,
            return_predictions=return_predictions,
        )
        if return_predictions:
            pred, coverage = out
            return adata, pred, coverage
        return adata

    return _run_on_adata_or_path(
        adata_or_path,
        model="baseline2",
        prefix=prefix,
        model_dir=model_dir,
        count_layer=count_layer,
        device=device,
        projection_batch_size=projection_batch_size,
        threshold=threshold,
        copy=copy,
        return_predictions=return_predictions,
    )


def baseline_tissue(
    adata_or_path: ad.AnnData | str | os.PathLike,
    *,
    model_dir: str | os.PathLike | None = None,
    count_layer: str | None = None,
    device: str = "cpu",
    projection_batch_size: int | None = None,
    threshold: float | None = None,
    prefix: str = "ZMAP_Tissue_prediction",
    copy: bool = False,
    return_predictions: bool = False,
):
    """
    Run the trained ZMAP/ensemble ``baseline2_tissue`` Tissue classifier.

    For AnnData input this follows the same output contract as
    ``predict_labels_kNN`` and writes ``ZMAP_Tissue_predicted`` plus metadata
    under ``adata.uns['zmap_labels']['ZMAP_Tissue']``. For H5AD path input it
    returns the lower-level predictions DataFrame.
    """
    if isinstance(adata_or_path, ad.AnnData):
        adata = adata_or_path.copy() if copy else adata_or_path
        out = predict_labels_mlp(
            adata,
            prediction_level="tissue",
            counts_source=count_layer,
            model_dir=model_dir,
            device=device,
            projection_batch_size=projection_batch_size,
            p_thresh=threshold if threshold is not None else 0.8,
            return_predictions=return_predictions,
        )
        if return_predictions:
            pred, coverage = out
            return adata, pred, coverage
        return adata

    return _run_on_adata_or_path(
        adata_or_path,
        model="baseline_tissue",
        prefix=prefix,
        model_dir=model_dir,
        count_layer=count_layer,
        device=device,
        projection_batch_size=projection_batch_size,
        threshold=threshold,
        copy=copy,
        return_predictions=return_predictions,
    )
