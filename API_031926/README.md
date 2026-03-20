# zmap-tools

[![Documentation](https://readthedocs.org/projects/zmap-tools/badge/?version=latest)](https://zmap-tools.readthedocs.io)
[![PyPI version](https://img.shields.io/pypi/v/zmap-tools.svg)](https://pypi.org/project/zmap-tools/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Python API for the **Zebrafish Multi-Atlas Project (ZMAP)** — a curated single-cell RNA-seq reference atlas for zebrafish development.

`zmap-tools` provides a simple **load → annotate → visualize** workflow for transferring cell-type labels from the ZMAP reference to your own datasets.

## Installation

```bash
pip install zmap-tools
```

Or from GitHub:

```bash
pip install git+https://github.com/WagnerLabUCSF/zmap-tools.git
```

## Quick start

### 1. Load the reference

```python
import zmap

# Download and cache the ZMAP reference (persists on Google Drive in Colab)
adata_ref = zmap.ref.load_zmap_h5ad()
```

Available presets: `"processed_slim_tpm"` (default, best for plotting), `"symphony"` (required for label transfer), `"processed"`, `"processed_slim"`, `"raw"`.

### 2. Annotate a query dataset

```python
zmap.predict.annotate_with_zmap(
    adata_query,
    query_raw_counts_source="counts",   # where your raw counts live
    cluster_col="leiden",               # your cluster column
)
```

This runs the full pipeline — TPM normalization, Symphony embedding, kNN label transfer, QC filtering, and plotting — in one call. Results land in `adata_query.obs`:

| Column | Description |
|---|---|
| `ZMAP_CellType_predicted` | Transferred cell-type label |
| `ZMAP_CellType_predicted_prob` | kNN vote probability (0–1) |
| `ZMAP_time_id` | Predicted developmental time (hpf) |

### 3. Visualize

```python
# Two-panel dotplot: gene expression across cell types × timepoints and studies
zmap.dotplot.gene_view(adata_ref, "sox2")

# Sibling comparison: a focal cell type vs. its relatives and tissues
zmap.dotplot.group_view(adata_ref, "hepatocyte")
```

### 4. Access consensus markers

```python
# Top markers per cell type as a dict
markers = zmap.ref.load_consensus_markers()

# As a panel DataFrame for dotplot input
panel = zmap.ref.load_consensus_markers(level="Tissue", n_per_group=10, format="panel")
```

Available levels: `"GermLayer"`, `"Tissue"`, `"CellType"`, `"CellTypeFine"`, `"Cluster"`, `"Leiden100"`.

## API overview

| Module | Purpose | Key functions |
|---|---|---|
| `zmap.ref` | Reference data | `load_zmap_h5ad()`, `load_consensus_markers()` |
| `zmap.predict` | Label transfer | `annotate_with_zmap()`, `predict_labels_kNN()`, `predict_labels_tissue_kNN()`, `aggregate_by_cluster()` |
| `zmap.dotplot` | Visualization | `gene_view()`, `group_view()`, `group_descendants_vs_markers()` |

## Workflow details

### Label transfer pipeline

`annotate_with_zmap` chains these steps (each also callable individually):

1. **Preprocess** — `preprocess_adata_query()`: TPM normalization + log1p
2. **Embed** — Symphony mapping into the ZMAP PCA/Harmony space
3. **Transfer** — `predict_labels_kNN()`: distance-weighted kNN voting with per-cell confidence scores
4. **Filter** — probability and distance thresholds flag low-confidence assignments
5. **Summarize** — `aggregate_by_cluster()`: cluster-level consensus calls with margin statistics
6. **Plot** — UMAP overlay with on-data labels + label-overlap heatmaps

### Dotplot modules

**Gene-centric** (`gene_view`): visualize one gene across all cell types, split by developmental timepoint (left panel) and study (right panel). Assesses both temporal dynamics and cross-study reproducibility.

**Group-centric** (`group_view`): given a focal cell type, fetches its consensus markers and plots a two-block dotplot — siblings sharing the same parent in the ZMAP hierarchy (top) and all tissues (bottom). Rows sorted by mean expression; support rings encode cross-study reproducibility.

**Descendant drilldown** (`group_descendants_vs_markers`): zoom into a parent group (e.g. a tissue) and show all child clusters with their own marker genes, optionally ordered by dendrogram.

## Data access

Reference H5ADs and consensus marker tables are hosted on Cloudflare R2 and downloaded on first use. Files are cached to Google Drive when mounted (`/content/drive/MyDrive/zmap/h5ad`), so they persist across Colab sessions. On local machines, caching falls back to `<cwd>/zmap/h5ad` and `~/.cache/zmap_tools`.

## Dependencies

Core: `anndata`, `scanpy`, `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `seaborn`, `tqdm`, `adjustText`, `symphonypy`

Requires Python ≥ 3.10.

## Documentation

Full API reference: [zmap-tools.readthedocs.io](https://zmap-tools.readthedocs.io)

## Citation

If you use `zmap-tools` in your work, please cite the ZMAP project (citation forthcoming).

## License

See [LICENSE](LICENSE).
