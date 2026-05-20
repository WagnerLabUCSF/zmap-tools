zmap-tools

## Environments

Use one FAISS variant per environment:

- CPU: `conda env create -f environment.cpu.yml`
- GPU: `conda env create -f environment.gpu.yml`
- Base (no FAISS): `conda env create -f environment.base.yml`

Do not install `faiss-cpu` and `faiss-gpu` into the same conda environment.

Then activate and install in editable mode (already included in yml via `-e .`):

- `conda activate zmap-api-cpu` or `conda activate zmap-api-gpu`

If you install with pip directly and want FAISS CPU auto-installed:

- `pip install -e ".[faiss-cpu]"`

## kNN Backend Inference

`zmap.predict.predict_labels_kNN` supports:

- `knn_backend`: `"auto" | "faiss" | "sklearn"` (default: `"auto"`)
- `knn_device`: `"auto" | "cpu" | "cuda" | "cuda:N"` (default: `"auto"`)

Behavior:

- `auto`: tries FAISS first, then falls back to sklearn if unavailable.
- `knn_device` is used only for FAISS.
- If CUDA is requested but unavailable, FAISS falls back to CPU.

Run metadata is stored in:

- `adata_query.uns["zmap_neighbors"]`:
  - `knn_backend_requested`, `knn_device_requested`
  - `knn_backend_used`, `knn_device_used`

## New High-Level Functions

- `zmap.apply_encoder(adata_query, ...)`
  - Projects query to latent using model resolution priority:
    - `ZMAP_ENCODER_MODEL_DIR` (env var)
    - `$ZMAP_HOME/models/phase3_normalize_noknn_fullsample_v2_cpu_fast_harmony_dropout0p2/model`
    - legacy local fallback path (for backward compatibility)
  - Writes latent to `adata_query.obsm["X_pca_harmony_pred"]` by default.

- `zmap.predict.predict_labels_kNN(adata_query, adata_ref, ...)`
  - Runs label transfer on latent space.
  - Default is ordinary kNN: `tissue_aware=False`.
  - Enable tissue-aware kNN with `tissue_aware=True`.
  - When `tissue_aware=True`, you must either:
    - pass `tissue_col="..."` for the tissue column present in reference and query, or
    - set `predict_tissue_level=True` to infer query tissue labels first as context.
  - Do not use `tissue_aware=True` when `ref_label_col` itself is the tissue column; for tissue-level prediction, use ordinary kNN.
  - Tissue-aware neighbor modes are selected with `tissue_mode="hard"` or `"soft"`.
  - Tissue-aware hard mode searches within matching tissue when enough reference cells exist, otherwise falls back to global kNN.
  - Tissue-aware soft mode adds `tissue_penalty_lambda` to mismatched-tissue neighbor distances before selecting neighbors.
  - Caches neighbor indices/distances in `adata_query.uns["zmap_neighbors"]` and includes tissue-aware settings in the cache key.
  - Writes outputs to query `obs`:
    - `<label_space>` or `<label_space>_<label_suffix>`
    - `<label_space>_unfilt`
    - `<label_space>_prob`
    - `<label_space>_dist`
  - Run summaries are stored under `adata_query.uns["zmap_labels"][label_space]`.

- `zmap.time_prediction(...)`
  - Runs kNN-based time regression using the same latent space and tissue-aware mode options.
  - Reuses `adata_query.uns["zmap_knn_cache"]` when settings match (avoids recomputing kNN).
  - Writes time outputs to query `obs` with `<time_col>_...` columns and stores metadata in
    `adata_query.uns["zmap_time_prediction"]`.

- `zmap.knn_config(...)`
  - Builds a shared KNN config dict for encoder-pipeline utilities such as
    `zmap.time_prediction(...)`.

- `zmap.plot_confusion_matrix(...)`
  - Plots confusion matrices with the legacy train/apply style.
  - Example:
    - `zmap.plot_confusion_matrix(adata_query, source="knn", level="ZMAP_CellType", normalize=False)`
    - `zmap.plot_confusion_matrix(adata_query, source="time", normalize=False)`

- `zmap.plot_leiden_celltype(...)`
  - Plots Leiden x predicted-celltype proportion heatmap directly from `adata_query.obs`.

- `zmap.plot_conf_entropy_margin(...)`
  - Plots confidence (x) vs entropy (y), colored by margin.
  - Supports threshold tuple `(conf, entropy, margin)`.
  - Points failing active thresholds are shown as `X`.
  - Example:
    - `zmap.plot_conf_entropy_margin(adata_query, base="ZMAP_CellType", threshold=(0, 0, 0.2))`

## Default Asset Directory

- `ZMAP_HOME` controls package asset directory.
- If unset, defaults to:
  - `$XDG_DATA_HOME/zmap` (if `XDG_DATA_HOME` is set), else
  - `~/.local/share/zmap`
